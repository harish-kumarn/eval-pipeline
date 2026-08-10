import ast
import json
import pandas as pd

# ---------------------------------------------------------
# 1. Load merged JSON
# ---------------------------------------------------------
with open("retriver_res/merged_testing_dataset.json", "r") as f:
    data = json.load(f)

# ---------------------------------------------------------
# 2. Build lookup:
#    question -> answer_chunk_id, tag, question_type
# ---------------------------------------------------------
question_lookup = {}

chunks_iter = data.values() if isinstance(data, dict) else data

for chunk in chunks_iter:
    questions = chunk.get("questions", {})

    for question_type in ("direct", "indirect"):
        for q_item in questions.get(question_type, []):

            question_lookup[q_item["question"].strip()] = {
                "answer_chunk_id": q_item["answer_chunk_id"],
                "tag": q_item["tag"],
                "question_type": question_type,
            }

# ---------------------------------------------------------
# 3. Load Retriever Results CSV
# ---------------------------------------------------------
df = pd.read_csv("retriver_res/04_08_dummy1_18_54.csv")

# ---------------------------------------------------------
# 4. Lookup metadata
# ---------------------------------------------------------
def get_question_metadata(query):
    return question_lookup.get(str(query).strip(), None)


metadata = df["query"].apply(get_question_metadata)

# ---------------------------------------------------------
# 5. Add Ground Truth Columns
# ---------------------------------------------------------
df["answer_chunk_id"] = metadata.apply(
    lambda x: x["answer_chunk_id"] if x else None
)

df["tag"] = metadata.apply(
    lambda x: x["tag"] if x else None
)

df["question_type"] = metadata.apply(
    lambda x: x["question_type"] if x else None
)

# ---------------------------------------------------------
# 6. Prediction
# True if ANY ground-truth chunk is retrieved
# ---------------------------------------------------------
def predicted(answer_chunk_ids, retrieved_chunks):

    if answer_chunk_ids is None:
        return False

    if pd.isna(retrieved_chunks):
        return False

    # Convert string representation of list to Python list
    if isinstance(retrieved_chunks, str):
        try:
            retrieved_chunks = ast.literal_eval(retrieved_chunks)
        except Exception:
            retrieved_chunks = [retrieved_chunks]

    return any(chunk in retrieved_chunks for chunk in answer_chunk_ids)


df["predicted"] = df.apply(
    lambda row: predicted(
        row["answer_chunk_id"],
        row["final_retrieved_chunks"],
    ),
    axis=1,
)

# ---------------------------------------------------------
# 7. Find Position of Retrieved Ground Truth Chunks
# Returns 1-based positions
# Example:
# answer_chunk_id = ["page_1_chunk_1","page_1_chunk_2"]
# retrieved_chunks =
# ["page_1_chunk_1","page_1_chunk_2","page_2_chunk_1"]
#
# Output:
# [1,2]
# ---------------------------------------------------------
def get_chunk_positions(answer_chunk_ids, retrieved_chunks):

    if answer_chunk_ids is None:
        return None

    if pd.isna(retrieved_chunks):
        return None

    if isinstance(retrieved_chunks, str):
        try:
            retrieved_chunks = ast.literal_eval(retrieved_chunks)
        except Exception:
            retrieved_chunks = [retrieved_chunks]

    positions = []

    for chunk in answer_chunk_ids:

        if chunk in retrieved_chunks:
            positions.append(retrieved_chunks.index(chunk) + 1)
        else:
            positions.append(None)

    return positions


df["retrieved_position"] = df.apply(
    lambda row: get_chunk_positions(
        row["answer_chunk_id"],
        row["final_retrieved_chunks"],
    ),
    axis=1,
)

# ---------------------------------------------------------
# 8. Save Results
# ---------------------------------------------------------
output_path = "retriver_res/04_08_dummy1_18_54_with_answers.csv"

df.to_csv(output_path, index=False)

print(f"Saved results to: {output_path}")

# ---------------------------------------------------------
# 9. Preview
# ---------------------------------------------------------
print(
    df[
        [
            "query",
            "answer_chunk_id",
            "tag",
            "question_type",
            "final_retrieved_chunks",
            "predicted",
            "retrieved_position",
        ]
    ].head()
)