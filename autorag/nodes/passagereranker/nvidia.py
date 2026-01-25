import os
from typing import List, Optional, Tuple

import aiohttp
import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import get_event_loop, process_batch, result_to_dataframe


class NvidiaReranker(BasePassageReranker):
	def __init__(self, project_dir: str, *args, **kwargs):
		"""
		Initialize Nvidia rerank node.

		:param project_dir: The project directory path.
		:param api_key: The API key for Nvidia rerank.
		    You can set it in the environment variable NVIDIA_API_KEY.
		    Or, you can directly set it on the config YAML file using this parameter.
		    Default is env variable "NVIDIA_API_KEY".
		:param kwargs: Extra arguments that are not affected
		"""
		super().__init__(project_dir)
		self.api_key = kwargs.pop("api_key", None)
		self.api_key = (
			os.getenv("NVIDIA_API_KEY", None) if self.api_key is None else self.api_key
		)
		if self.api_key is None:
			raise KeyError(
				"Please set the API key for Nvidia rerank in the environment variable NVIDIA_API_KEY "
				"or directly set it on the config YAML file."
			)
		self.invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
		self.session = aiohttp.ClientSession(loop=get_event_loop())
		self.session.headers.update(
			{"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
		)

	def __del__(self):
		if hasattr(self, "session"):
			if not self.session.closed:
				loop = get_event_loop()
				if loop.is_running():
					loop.create_task(self.session.close())
				else:
					loop.run_until_complete(self.session.close())
			del self.session
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, scores, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		batch = kwargs.pop("batch", 64)
		model = kwargs.pop("model", "nvidia/rerank-qa-mistral-4b")
		truncate = kwargs.pop("truncate", None)
		return self._pure(queries, contents, scores, ids, top_k, batch, model, truncate)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 64,
		model: str = "nvidia/rerank-qa-mistral-4b",
		truncate: Optional[str] = None,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents with Nvidia rerank models.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param scores_list: The list of lists of scores retrieved from the initial ranking
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
		:param model: The model name for Nvidia rerank.
		    Default is "nvidia/rerank-qa-mistral-4b".
		:param truncate: Optional truncation strategy for the API request
		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		tasks = [
			nvidia_rerank_pure(
				self.session,
				self.invoke_url,
				model,
				query,
				document,
				ids,
				top_k,
				truncate=truncate,
			)
			for query, document, ids in zip(queries, contents_list, ids_list)
		]
		loop = get_event_loop()
		results = loop.run_until_complete(process_batch(tasks, batch_size=batch))

		content_result = list(map(lambda x: x[0], results))
		id_result = list(map(lambda x: x[1], results))
		score_result = list(map(lambda x: x[2], results))

		return content_result, id_result, score_result


async def nvidia_rerank_pure(
	session: aiohttp.ClientSession,
	invoke_url: str,
	model: str,
	query: str,
	documents: List[str],
	ids: List[str],
	top_k: int,
	truncate: Optional[str] = None,
) -> Tuple[List[str], List[str], List[float]]:
	"""
	Async function to call Nvidia Rerank API.

	:param session: The aiohttp session to use for reranking
	:param invoke_url: The Nvidia Rerank API endpoint
	:param model: The model name for Nvidia rerank
	:param query: The query to use for reranking
	:param documents: The list of contents to rerank
	:param ids: The list of ids corresponding to the documents
	:param top_k: The number of passages to be retrieved
	:param truncate: Optional truncation strategy for the API request
	:return: Tuple of lists containing the reranked contents, ids, and scores
	"""
	payload = {
		"model": model,
		"query": {"text": query},
		"passages": [{"text": doc} for doc in documents],
	}
	if truncate is not None:
		payload["truncate"] = truncate

	async with session.post(invoke_url, json=payload) as response:
		if response.status != 200:
			raise ValueError(
				f"NVIDIA API Error: {response.status} - {await response.text()}"
			)

		response_body = await response.json()

		rankings = response_body.get("rankings", [])

		def _score(item):
			if item.get("logit") is not None:
				return float(item["logit"])
			if item.get("score") is not None:
				return float(item["score"])
			return 0.0

		rankings.sort(key=_score, reverse=True)

		top_rankings = rankings[:top_k]

		reranked_contents = []
		reranked_ids = []
		reranked_scores = []

		for item in top_rankings:
			idx = item["index"]
			reranked_contents.append(documents[idx])
			reranked_ids.append(ids[idx])
			reranked_scores.append(_score(item))

		return reranked_contents, reranked_ids, reranked_scores
