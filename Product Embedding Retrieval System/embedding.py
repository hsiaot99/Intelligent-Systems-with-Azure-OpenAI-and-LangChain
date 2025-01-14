from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models, PointStruct
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained embedding model
try:
    model = SentenceTransformer('all-mpnet-base-v2')
    logging.info("Embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading embedding model: {e}")
    exit(1)


# Connect to Qdrant.  Replace with your connection string for persistent storage.
try:
    client = QdrantClient(":memory:")
    logging.info("Connected to Qdrant.")
except Exception as e:
    logging.error(f"Error connecting to Qdrant: {e}")
    exit(1)


# Product descriptions (from user input)
product_descriptions = {
    "A": "電子零件 A - 微型處理器 型號: A100 功能: 高效能低功耗微處理器，適用於嵌入式系統。 主要特點: 核心數量: 4 最大時鐘速度: 1.5 GHz 支持 SIMD 指令集 功耗: 10W 應用領域: 物聯網(IoT)裝置、智能家居系統、可穿戴技術。",
    "B": "電子零件 B - 高頻率晶體振盪器 型號: B200 功能: 提供穩定的時鐘信號，用於高精度時序控制。 主要特點: 頻率範圍: 10 MHz 至 100 MHz 溫度穩定性: ±2 ppm 低相位雜訊 應用領域: 通信設備、高速數據處理、精密測量儀器。",
    "C": "電子零件 C - 多層陶瓷電容 (MLCC) 型號: C300 功能: 在電路中提供穩定的電容，用於濾波、去耦和能量存儲。 主要特點: 電容範圍: 1 nF 至 100 μF 額定電壓: 6.3V 至 50V 尺寸小，容量大 應用領域: 電源管理、信號處理、消費電子產品。",
    "D": "電子零件 D - 功率晶體管 型號: D400 功能: 控制高電流負載，用於放大和開關應用。 主要特點: 最大集電極電流: 20A 最大耗散功率: 100W 低飽和壓降 應用領域: 電源轉換、電動車、可再生能源系統。",
    "E": "電子零件 E - 光學感測器 型號: E500 功能: 檢測光線強度和顏色，用於自動亮度調整和顏色識別。 主要特點: 靈敏度範圍: 400 nm 至 700 nm 數字輸出，簡化接口 低功耗設計 應用領域: 智能照明系統、顯示屏技術、安全和監控系統。"
}


# Generate embeddings and store in Qdrant
try:
    client.recreate_collection(
        collection_name="products",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.DOT),
    )
    logging.info("Qdrant collection 'products' recreated.")
    points = []
    for product_id, description in product_descriptions.items():
        try:
            embedding = model.encode(description)
            points.append(
                PointStruct(
                    id=int(product_id.encode("utf-8").hex(), 16),
                    vector=embedding.tolist(),
                    payload={"product_id": product_id, "description": description},
                )
            )
            logging.info(f"Embedding generated for product {product_id}.")
        except Exception as e:
            logging.error(f"Error generating embedding for product {product_id}: {e}")
    client.upsert(collection_name="products", points=points)
    logging.info("Embeddings successfully stored in Qdrant.")
except Exception as e:
    logging.error(f"An error occurred: {e}")
    exit(1)
