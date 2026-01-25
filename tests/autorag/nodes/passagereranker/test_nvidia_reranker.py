from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses

from autorag.nodes.passagereranker import NvidiaReranker
from autorag.nodes.passagereranker.nvidia import nvidia_rerank_pure
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	scores_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)

NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"


@pytest.mark.asyncio()
async def test_nvidia_rerank_pure():
	with aioresponses() as m:
		mock_response = {
			"rankings": [
				{"index": 1, "logit": 0.9},
				{"index": 0, "logit": 0.2},
			]
		}
		m.post(NVIDIA_RERANK_URL, payload=mock_response)
		async with aiohttp.ClientSession() as session:
			session.headers.update(
				{"Authorization": "Bearer mock_api_key", "Accept": "application/json"}
			)
			content_result, id_result, score_result = await nvidia_rerank_pure(
				session,
				NVIDIA_RERANK_URL,
				"nvidia/rerank-qa-mistral-4b",
				queries_example[0],
				contents_example[0],
				ids_example[0],
				top_k=2,
			)
		assert len(content_result) == 2
		assert len(id_result) == 2
		assert len(score_result) == 2

		assert all([res in contents_example[0] for res in content_result])
		assert all([res in ids_example[0] for res in id_result])

		assert score_result[0] >= score_result[1]


async def mock_nvidia_rerank_pure(
	session, invoke_url, model, query, documents, ids, top_k, truncate=None
):
	if query == queries_example[0]:
		return (
			[documents[1], documents[2], documents[0]][:top_k],
			[ids[1], ids[2], ids[0]][:top_k],
			[0.8, 0.2, 0.1][:top_k],
		)
	if query == queries_example[1]:
		return (
			[documents[1], documents[0], documents[2]][:top_k],
			[ids[1], ids[0], ids[2]][:top_k],
			[0.8, 0.2, 0.1][:top_k],
		)
	raise ValueError(f"Unexpected query: {query}")


@pytest.fixture
def nvidia_reranker_instance():
	return NvidiaReranker(project_dir=project_dir, api_key="test")


@patch(
	"autorag.nodes.passagereranker.nvidia.nvidia_rerank_pure",
	mock_nvidia_rerank_pure,
)
def test_nvidia_reranker(nvidia_reranker_instance):
	top_k = 3
	contents_result, id_result, score_result = nvidia_reranker_instance._pure(
		queries_example, contents_example, scores_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch(
	"autorag.nodes.passagereranker.nvidia.nvidia_rerank_pure",
	mock_nvidia_rerank_pure,
)
def test_nvidia_reranker_batch_one(nvidia_reranker_instance):
	top_k = 3
	batch = 1
	contents_result, id_result, score_result = nvidia_reranker_instance._pure(
		queries_example,
		contents_example,
		scores_example,
		ids_example,
		top_k,
		batch=batch,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch(
	"autorag.nodes.passagereranker.nvidia.nvidia_rerank_pure",
	mock_nvidia_rerank_pure,
)
def test_nvidia_reranker_node():
	top_k = 1
	result_df = NvidiaReranker.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		top_k=top_k,
		api_key="test",
	)
	base_reranker_node_test(result_df, top_k)
