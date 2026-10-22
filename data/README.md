# Data

USERPRINT uses two public datasets, which are not redistributed here:

| Role | Dataset | Use |
|---|---|---|
| Persona source | **LMSYS Chatbot Arena** conversations | User utterances are selected, embedded and clustered into personas |
| Transfer target | **PersonaChat** (Zhang et al., 2018; Facebook AI) | 300 dialogues are answered under the three conditioning set-ups |

The paper's released benchmark, with the generated responses, persona clusters and evaluation annotations, is on Hugging Face: [abittolu/UserPrint](https://huggingface.co/datasets/abittolu/UserPrint).

## Formats

**Conversations** (`userprint.select --input`): JSONL in the LMSYS Chatbot Arena format. Each row has a `conversation_a` (or `conversation` / `messages`) list of `{"role": "user" | "assistant", "content": ...}`, or a plain `text` field.

**Dialogues** (`userprint.generate --dialogues`): JSONL with one PersonaChat dialogue per row:

```json
{"id": "d1", "history": ["hi!", "hello, how are you?", "tired after my run, any tips?"], "reference": "stretch and drink water."}
```

`history` alternates between the two speakers and ends with the turn to be answered. `reference` is the gold next turn used for ROUGE-L and BERTScore.

## Sample

[`sample/`](sample/) holds a few **invented** conversations and dialogues in these formats, so the pipeline and tests run offline. They are not from LMSYS or PersonaChat.
