from qdrant_client import QdrantClient, models
import json

# Connect to Qdrant
client = QdrantClient(":memory:") # Use in-memory for this example. For persistent storage, use a proper connection string.

# Product to retrieve
product_id = "A"

try:
    # Search for the product
    search_result = client.search(
        collection_name="products",
        query_vector=[0] * 768,  # Placeholder vector, replace with actual vector if needed.
        query_filter=models.Filter(must=[models.FieldCondition(key="product_id", match=models.MatchValue(value=product_id))]),
        limit=1,
    )

    # Print the results
    if search_result:
        print(json.dumps(search_result[0].payload, indent=2))
    else:
        print(f"Product with ID '{product_id}' not found.")

except Exception as e:
    print(f"An error occurred: {e}")
