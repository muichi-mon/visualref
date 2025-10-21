import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

import faiss
from src.models.configs import get_model_config
from src.config import resolve_repo
from src.models.llava import init_llava
from src.models.relevance_feedback import (
    CaptionVLMRelevanceFeedback,
    ImageBasedVLMRelevanceFeedback,
    RocchioUpdate,
)
from src.utils.image_utils import resize_images


class RetrievalService:
    def __init__(
        self,
        config: Dict[str, Any],
        faiss_index: str,
        captioning_model_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
    ):
        self.config = config
        self.captioning_model_config = captioning_model_config if captioning_model_config is not None else None
        self.faiss_index = faiss_index
        self.accumulated_query_embeddings = {"query_embedding": None}
        self.retrieval_round = 1
        self.experiment_id = 0
        self.device = device
        
        self._init_backbone()
        if self.captioning_model_config is not None:
            self._init_captioning_model()
            self._init_captioning_relevance_feedback()
        self._init_rocchio_update(alpha=alpha, beta=beta, gamma=gamma)
        self._init_faiss_index()

    def _init_backbone(self):
        self.backbone_config = get_model_config(
            self.config["VLM_MODEL_FAMILY"],
            self.config["VLM_MODEL_NAME"]
        )
        self.backbone = self.backbone_config["model_class"].from_pretrained(self.config["VLM_MODEL_NAME"])
        self.backbone.eval()
        self.backbone_processor = (
            self.backbone_config["processor_class"]
            .from_pretrained(self.config["VLM_MODEL_NAME"])
        )

        self.wrapper = self.backbone_config["wrapper_class"](
            model=self.backbone,
            processor=self.backbone_processor
        )
    
    def _init_captioning_model(self):
        model_config = get_model_config(
            self.captioning_model_config["MODEL_FAMILY"], 
            self.captioning_model_config["MODEL_ID"]
        )
        if self.captioning_model_config["MODEL_FAMILY"] == "llava":
            self.captioning_model = init_llava(
                model_config=model_config,
                device=self.device,
                use_8bit=self.captioning_model_config["USE_8BIT"]
            )
        else:
            raise ValueError(
                f"Captioning model family {self.captioning_model_config['model_family']} not supported"
            )

    def _init_captioning_relevance_feedback(self):
        self.captioning_relevance_feedback = CaptionVLMRelevanceFeedback(
            vlm_wrapper_retrieval=self.wrapper,
            vlm_wrapper_captioning=self.captioning_model,
        )

    def _init_rocchio_update(
        self,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        multiple: bool = False,
    ):
        self.rocchio_update = RocchioUpdate(alpha=alpha, beta=beta, gamma=gamma)

    def _init_faiss_index(self):
        # Ensure FAISS receives a plain string path and validate existence
        index_path_str = str(self.faiss_index)
        if not os.path.exists(index_path_str):
            raise ValueError(
                f"FAISS index file not found at '{index_path_str}'. "
                f"Verify APP_INDEX_PATH/INDEX_PATH and volume mounts."
            )
        try:
            self.index = faiss.read_index(index_path_str)
        except RuntimeError as e:
            raise ValueError(f"Failed to read FAISS index: {e}. Check if the index file exists.")
        try:
            with open(
                os.path.join(os.path.dirname(index_path_str),
                "image_paths.txt"),
                "r"
            ) as f:
                self.candidate_image_paths = [line.strip() for line in f.readlines()]
            # Normalize candidate paths to absolute repo-root-based paths
            self.candidate_image_paths = [resolve_repo(p) for p in self.candidate_image_paths]
        except FileNotFoundError as e:
            raise ValueError(f"Failed to read image paths: {e}. Check if the image paths file exists.")

    def search_images(self, query: str, top_k: int = 5):
        """Extract image_search function logic"""
        self.experiment_id += 1

        processed_query = self.wrapper.process_inputs(text=query)
        with torch.no_grad():
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

        self.accumulated_query_embeddings["query_embedding"] = query_embedding

        scores, img_ids = self.index.search(query_embedding, top_k)
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        print(retrieved_image_paths)
        for path in retrieved_image_paths:
            assert os.path.exists(path), f"Image path {path} does not exist"
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        print(retrieved_images)
        retrieved_images = resize_images(
            retrieved_images,
            (self.config.get("IMG_SIZE", 224), self.config.get("IMG_SIZE", 224))
        )

        return retrieved_images, scores, retrieved_image_paths

    def process_feedback(
        self,
        query: str,
        relevant_image_paths: List[str],
        user_prompt: Optional[str] = None,
        annotator_json_boxes_list: Optional[List[Any]] = None,
        visualization: bool = False,
        top_k_feedback: int = 5,
        prompt_based_on_query: bool = False,
        relevant_captions: Optional[Union[List[str], str]] = None,
        irrelevant_captions: Optional[Union[List[str], str]] = None,
        prompt: Optional[str] = None
    ):
        relevance_feedback_results = self.captioning_relevance_feedback(
            query=query,
            relevant_image_paths=relevant_image_paths,
            user_prompt=user_prompt,
            visualization=visualization,
            top_k_feedback=top_k_feedback,
            annotator_json_boxes_list=annotator_json_boxes_list,
            prompt_based_on_query=prompt_based_on_query,
            relevant_captions=relevant_captions,
            irrelevant_captions=irrelevant_captions,
            prompt=prompt
        )

        return {
            "positive": relevance_feedback_results["positive"].tolist() if relevance_feedback_results["positive"] is not None else None,
            "negative": relevance_feedback_results["negative"].tolist() if relevance_feedback_results["negative"] is not None else None,
            "relevant_captions": relevance_feedback_results["relevant_captions"],
            "irrelevant_captions": relevance_feedback_results["irrelevant_captions"],
            "explanation": relevance_feedback_results["explanation"]
        }

    def apply_feedback(
        self,
        query: str,
        top_k: int,
        relevant_captions: Optional[Union[List[str], torch.Tensor]] = None,
        irrelevant_captions: Optional[Union[List[str], torch.Tensor]] = None,
        fuse_initial_query: bool = False
    ):
        """Extract feedback_loop function logic"""
        processed_query = self.wrapper.process_inputs(text=query)
        with torch.no_grad():
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

        rocchio_query_embedding = (self.accumulated_query_embeddings["query_embedding"] + query_embedding) / 2 if (
            fuse_initial_query
        ) else self.accumulated_query_embeddings["query_embedding"]

        relevant_captions = [cap for cap in relevant_captions if cap != ""]
        irrelevant_captions = [cap for cap in irrelevant_captions if cap != ""]

        print(relevant_captions, irrelevant_captions)

        with torch.no_grad():
            if relevant_captions is not None and relevant_captions:
                positive_embeddings = self.wrapper.get_text_embeddings(
                        self.wrapper.process_inputs(text=relevant_captions)
                    )
                positive_embeddings = positive_embeddings.mean(dim=0)
            else:
                positive_embeddings = None
            if irrelevant_captions is not None and irrelevant_captions:
                negative_embeddings = self.wrapper.get_text_embeddings(
                self.wrapper.process_inputs(text=irrelevant_captions)
            )
                negative_embeddings = negative_embeddings.mean(dim=0)
            else:
                negative_embeddings = None

        self.accumulated_query_embeddings["query_embedding"] = self.rocchio_update(
            query_embeddings=rocchio_query_embedding,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings
        )

        scores, img_ids = self.index.search(self.accumulated_query_embeddings["query_embedding"], top_k)
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        retrieved_images = resize_images(
            retrieved_images,
            (self.config.get("IMG_SIZE", 224), self.config.get("IMG_SIZE", 224))
        )

        self.retrieval_round += 1
        return retrieved_images, scores, retrieved_image_paths


class RetrievalServiceVisual(RetrievalService):
    def __init__(
        self,
        config: Dict[str, Any],
        faiss_index: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
    ):  
        super().__init__(
            config=config,
            faiss_index=faiss_index,
            device=device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self._init_image_based_relevance_feedback()

    def _init_image_based_relevance_feedback(self):
        self.image_based_relevance_feedback = ImageBasedVLMRelevanceFeedback(
            vlm_wrapper_retrieval=self.wrapper,
        )

    def process_and_apply_feedback(
        self,
        query: str,
        top_k: int,
        relevant_image_paths: List[str],
        relevant_captions: Optional[str] = None,
        irrelevant_captions: Optional[str] = None,
        annotator_json_boxes_list: Optional[List[Any]] = None,
        fuse_initial_query: bool = False,
    ):
        relevance_feedback_results = self.image_based_relevance_feedback(
            query=query,
            relevant_image_paths=relevant_image_paths,
            annotator_json_boxes_list=annotator_json_boxes_list,
            top_k_feedback=top_k
        )

        relevant_segments = relevance_feedback_results["relevant_segments"]
        irrelevant_segments = relevance_feedback_results["irrelevant_segments"]

        with torch.no_grad():
            # Encode positive image segments into embeddings
            if relevant_segments is not None and relevant_segments:
                positive_image_embeddings = self.wrapper.get_image_embeddings(
                    self.wrapper.process_inputs(images=relevant_segments)
                )
                positive_image_embeddings = positive_image_embeddings.mean(dim=0)
            else:
                positive_image_embeddings = None
            # Encode negative image segments into embeddings
            if irrelevant_segments is not None and irrelevant_segments:
                negative_image_embeddings = self.wrapper.get_image_embeddings(
                    self.wrapper.process_inputs(images=irrelevant_segments)
                )
                negative_image_embeddings = negative_image_embeddings.mean(dim=0)
            else:
                negative_image_embeddings = None
            # Encode positive text captions into embeddings
            if relevant_captions is not None and relevant_captions:
                positive_text_embeddings = self.wrapper.get_text_embeddings(
                    self.wrapper.process_inputs(text=relevant_captions)
                )
                positive_text_embeddings = positive_text_embeddings.mean(dim=0)
            else:
                positive_text_embeddings = None
            # Encode negative text captions into embeddings
            if irrelevant_captions is not None and irrelevant_captions:
                negative_text_embeddings = self.wrapper.get_text_embeddings(
                    self.wrapper.process_inputs(text=irrelevant_captions)
                )
                negative_text_embeddings = negative_text_embeddings.mean(dim=0)
            else:
                negative_text_embeddings = None

            # Combine positive image embeddings and text embeddings
            if positive_image_embeddings is not None and positive_text_embeddings is not None:
                positive_embeddings = (positive_image_embeddings + positive_text_embeddings) / 2
            elif positive_image_embeddings is not None:
                positive_embeddings = positive_image_embeddings
            elif positive_text_embeddings is not None:
                positive_embeddings = positive_text_embeddings
            else:
                positive_embeddings = None
            # Combine negative image embeddings and text embeddings
            if negative_image_embeddings is not None and negative_text_embeddings is not None:
                negative_embeddings = (negative_image_embeddings + negative_text_embeddings) / 2
            elif negative_image_embeddings is not None:
                negative_embeddings = negative_image_embeddings
            elif negative_text_embeddings is not None:
                negative_embeddings = negative_text_embeddings
            else:
                negative_embeddings = None

            processed_query = self.wrapper.process_inputs(text=query)
            with torch.no_grad():
                query_embedding = self.wrapper.get_text_embeddings(processed_query)

            print("Fusing initial query embedding!")
            rocchio_query_embedding = (self.accumulated_query_embeddings["query_embedding"] + query_embedding) / 2 if (
                fuse_initial_query
            ) else self.accumulated_query_embeddings["query_embedding"]

            self.accumulated_query_embeddings["query_embedding"] = self.rocchio_update(
                query_embeddings=rocchio_query_embedding,
                positive_embeddings=positive_embeddings,
                negative_embeddings=negative_embeddings,
            )

        scores, img_ids = self.index.search(
            self.accumulated_query_embeddings["query_embedding"],
            top_k
        )
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        retrieved_images = resize_images(
            retrieved_images,
            (self.config.get("IMG_SIZE", 224), self.config.get("IMG_SIZE", 224))
        )

        self.retrieval_round += 1

        return retrieved_images, scores, retrieved_image_paths
