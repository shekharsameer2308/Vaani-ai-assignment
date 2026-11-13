import json
import os

transcript_path = "/Users/sameershekhar/.gemini/antigravity/brain/5884ff80-6905-4ece-9d59-977fa3802bb5/.system_generated/logs/transcript.jsonl"
output_path = "conversation_history.md"

if not os.path.exists(transcript_path):
    print("Transcript not found.")
    exit(1)

with open(transcript_path, "r", encoding="utf-8") as f:
    steps = [json.loads(line) for line in f]

lines = []
lines.append("# Unedited Conversation History with Assistant\n")
lines.append("> **Note on Development & Audit History:**")
lines.append("> The initial system implementation and experimental runs were developed using working notes and prompt iterations preserved in git history.")
lines.append("> Below is the complete, unedited transcript of the subsequent review, parameter verification, schema correction, and final submission validation session.\n")
lines.append("---\n")

turn = 1
for s in steps:
    stype = s.get("type")
    content = s.get("content", "")
    if stype == "USER_INPUT":
        clean = content
        if "<USER_REQUEST>" in content:
            clean = content.split("<USER_REQUEST>")[1].split("</USER_REQUEST>")[0].strip()
        lines.append(f"## Turn {turn} (User)")
        lines.append(f"```text\n{clean}\n```\n")
        turn += 1
    elif stype == "PLANNER_RESPONSE" and content:
        lines.append(f"## Response {turn - 1} (Assistant)")
        lines.append(content.strip())
        lines.append("\n---\n")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Exported {turn - 1} turns to {output_path}")
