import json
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from chatbot_graph import app
from langchain_core.messages import HumanMessage

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data/evaluation_set.json")

def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found details.")
        return

    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    results = []
    total_latency = 0
    correct_count = 0
    total_recall = 0
    total_precision = 0

    print(f"Starting evaluation on {len(dataset)} items...")

    for item in dataset:
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        
        start_time = time.time()
        
        # Invoke graph
        # We need a fresh state or handle context. For eval, let's treat each as new.
        state = {
            "messages": [HumanMessage(content=question)],
            "user_info": {},
            "dialog_stage": "general",
            "reservation_details": {}
        }
        
        response = app.invoke(state)
        end_time = time.time()
        
        latency = end_time - start_time
        total_latency += latency
        
        bot_response = response["messages"][-1].content.lower()
        retrieved_docs = response.get("retrieved_docs", [])
        
        # 1. Answer Accuracy (End-to-End)
        hit = any(k.lower() in bot_response for k in expected_keywords)
        if hit:
            correct_count += 1
            
        # 2. Retrieval Recall (Did retrieved docs contain expected keywords?)
        # Recall = (Keywords found in context) / (Total expected keywords)
        keywords_in_context = 0
        joined_context = " ".join(retrieved_docs).lower()
        for k in expected_keywords:
            if k.lower() in joined_context:
                keywords_in_context += 1
        recall = keywords_in_context / len(expected_keywords) if expected_keywords else 0
        total_recall += recall
        
        # 3. Context Precision (What fraction of chunks were relevant?)
        # Precision = (Chunks with at least 1 keyword) / (Total chunks)
        relevant_chunks = 0
        for doc in retrieved_docs:
            if any(k.lower() in doc.lower() for k in expected_keywords):
                relevant_chunks += 1
        precision = relevant_chunks / len(retrieved_docs) if retrieved_docs else 0
        total_precision += precision

        results.append({
            "question": question,
            "response": bot_response,
            "hit": hit,
            "recall": recall,
            "precision": precision,
            "latency": latency
        })

    avg_latency = total_latency / len(dataset) if dataset else 0
    accuracy = correct_count / len(dataset) if dataset else 0
    avg_recall = total_recall / len(dataset) if dataset else 0
    avg_precision = total_precision / len(dataset) if dataset else 0
    
    report = f"""
    Evaluation Report
    -----------------
    Total Queries: {len(dataset)}
        Answer Accuracy: {accuracy:.2f}
    Retrieval Recall: {avg_recall:.2f}
    Context Precision: {avg_precision:.2f}
    Average Latency: {avg_latency:.4f}s
    """
    
    print(report)
    
    # Save report
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
        f.write("\n\nDetails:\n")
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()
