
def populate_visionrag_results(self, log_path: Path) -> None:
    """Construct visionragresult.json from golden dataset, generated answers and retrieved chunks."""

    logger.info(f"Populating results from golden dataset, answers and log: {log_path}")

    print("\n========== STEP 1 ==========")
    print("Log file:", log_path)

    log_data = self._parse_retrieval_log(log_path)

    print("\n========== STEP 2 ==========")
    print("Parsed log_data:")
    print(log_data)

    all_results = self._merge_visionrag_results(self.questions, log_data)

    print("\n========== STEP 3 ==========")
    print("Merged Results:")
    print(all_results)

    if all_results:
        FileHandler.save_json(all_results, self.visionrag_file)
        logger.info(
            "Successfully populated %d entries into %s",
            len(all_results),
            self.visionrag_file,
        )
    else:
        logger.warning("No entries were constructed for visionrag result.")


fmt = CONFIG.get("chunk_format", "chunk{id}")

print("\nChunk format:", fmt)

with open(log_path, "r", encoding="utf-8") as f:

    for row in csv.DictReader(f):

        print("\n--------------------------------")
        print("CSV Row:")
        print(row)

        query = row.get("query", "").strip()
        chunks_str = row.get("final_retrieved_chunks", "")
        answer = row.get("answer", "").strip()

        print("Query:", query)
        print("Chunks String:", chunks_str)
        print("Answer:", answer)

        if not query:
            print("Empty query. Skipping row.")
            continue

        retrieved_chunks = []

        if chunks_str:
            try:
                chunks = ast.literal_eval(chunks_str)

                print("Parsed chunks:", chunks)
                print("Type:", type(chunks))

                if isinstance(chunks, list):

                    retrieved_chunks = [
                        fmt.format(id=cid)
                        for cid in chunks
                    ]

                    print("Retrieved chunks after formatting:")
                    print(retrieved_chunks)

            except (ValueError, SyntaxError):
                logger.info(
                    "Failed to parse chunks string: %.120s",
                    chunks_str
                )

        log_data[query] = {
            "retrieved_chunk_ids": retrieved_chunks,
            "generated_answer": answer,
        }

        print("Current log_data entry:")
        print(log_data[query])

print("\nFinal log_data:")
print(log_data)

print("\n========== FIND BEST MATCH ==========")
print("Question:", question)

keys = list(log_data.keys())

print("Available Keys:")
print(keys)

if question.strip() in log_data:
    print("Exact Match Found")
    return log_data[question.strip()]

matches = get_close_matches(
    question,
    keys,
    n=1,
    cutoff=0.3
)

print("Fuzzy Matches:", matches)

if matches:
    print("Matched with:", matches[0])
    return log_data[matches[0]]

print("No Match Found")
return {}


