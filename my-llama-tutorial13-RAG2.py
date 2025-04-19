import os, sys
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank
from llama_index.core import (
        SimpleDirectoryReader, 
        VectorStoreIndex, 
        Settings, 
        StorageContext, 
        load_index_from_storage,
        )
from llama_index.core.workflow import (
        Context,
        Workflow,
        StartEvent,
        StopEvent,
        step,
        )

Settings.llm = Ollama(model="granite3-dense", temperature=0, verbose =True)
#Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""
    nodes: list[NodeWithScore]


class RAGWorkflow(Workflow):
    @step
    async def retrieve(
            self, ctx: Context, ev: StartEvent
        ) -> StopEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        index = ev.get("index")

        if not query or not index:
            return StopEvent(result=None)

        print(f"Query the database with: {query}")

        # store the query in the global context
        #await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return StopEvent(result=None)

        rerank = SentenceTransformerRerank(
                top_n=3,
                model="cross-encoder/ms-macro-MiniLM-L-2-v2",
                keep_retrieval_score=True,
                )
        query_engine = index.as_query_engine(
                similarity_top_k=3,
                node_postprocessors=[rerank],
                )
        response = query_engine.query(query)
        return StopEvent(result=response)

async def ingestRAG(data_dir: str, index_dir: str) -> VectorStoreIndex:

        if os.path.exists(index_dir+"/docstore.json"):
            print("loading index from local disk...")
            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            index = load_index_from_storage(storage_context)
            return index

        elif os.path.exists(data_dir):
            print("embedding and indexing...")
            storage_context = StorageContext.from_defaults()
            documents = SimpleDirectoryReader(data_dir).load_data()
            index = VectorStoreIndex.from_documents(
                    documents=documents,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model,
                    )
            index.storage_context.persist(persist_dir=index_dir)
            return index
        else:
            print("Error: no index or data source available...")
            return None

async def main():
    index = await ingestRAG(data_dir="~/data", 
                       index_dir="~/storage")

    if index==None:
        print("quit")
        return None

    w = RAGWorkflow()

    while True:
        myquery = input("User: ")
        if myquery == "exit" or myquery == "quit" or myquery == "bye":
            break
        # Run a query
        result = await w.run(query=myquery,index=index)
        print("w.run output:", result)
        async for chunk in result.async_response_gen():
            print(chunk, end="", flush=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
