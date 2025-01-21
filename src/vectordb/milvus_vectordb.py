from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
import numpy as np
from typing import List, Optional, Dict, Any

class MilvusVectorDB:
    def __init__(self, collection_name: str, dim: int):
        self.collection_name = collection_name
        self.dim = dim  # OpenAI embedding dimension
        self.connect()
        self._create_collection()

    def connect(self):
        try:
            connections.connect(
                alias="default",
                host="localhost",
                port="19530",
                timeout=10  # Add timeout
            )
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    def _create_collection(self):
        if self.collection_name not in utility.list_collections():
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.JSON)  # Add metadata field
            ]
            schema = CollectionSchema(fields=fields)
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index for vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.collection = Collection(self.collection_name)
    

    def add_documents(self, texts, embeddings, batch_size=1):
        try:
            # Convert embeddings to numpy array if needed
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Prepare data in batches
            total_docs = len(texts)
            for i in range(0, total_docs, batch_size):
                end_idx = min(i + batch_size, total_docs)
                batch_texts = texts[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                
                entities = [
                    batch_texts,
                    batch_embeddings.tolist(),
                    [{}] * len(batch_texts)  # Add empty metadata for each document
                ]
                self.collection.insert(entities)
                print(f"Inserted {len(batch_texts)} documents")

            # Flush after batch insertion
            self.collection.flush()
            return True
            
        except Exception as e:
            print(f"Error inserting documents: {e}")
            return False

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors and return top k results
        """
        self.collection.load()
        
        # search_params = {
        #     "metric_type": "COSINE",
        #     "params": {"nprobe": 10}
        # }

        expr = None
        if metadata_filter:
            # Convert metadata filter to Milvus expression
            expr = " and ".join([f"metadata['{k}'] == '{v}'" for k, v in metadata_filter.items()])

        # results = self.collection.search(
        #     data=[query_vector],
        #     anns_field="embedding",
        #     param=search_params,
        #     limit=k,
        #     expr=expr,
        #     output_fields=["text", "metadata"]
        # )

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=k,
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "text": hit.entity.get("text"),
                "metadata": hit.entity.get("metadata"),
                "score": hit.score,
                "id": hit.id
            })

        return hits

    def delete_collection(self) -> None:
        """Delete the entire collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

    def __del__(self):
        """Cleanup connection when object is destroyed"""
        try:
            connections.disconnect(alias="default")
        except:
            pass



if __name__ == "__main__":
    #test
    vdb = MilvusVectorDB("test_collection", dim=3)
    vdb.add_documents(["Hello, world!", "Hello, world!", "Hello, world!"], 
                     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(vdb.similarity_search([1, 2, 3]))
