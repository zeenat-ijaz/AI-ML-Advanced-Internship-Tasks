# Task 5: Auto Tagging Support Tickets Using LLM

## Objective
Automatically tag free-text support tickets into categories using a large language model,
outputting the top 3 most probable tags per ticket.

## Dataset
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
(61,765 real-world-style support tickets with subject, body, and ground-truth `queue` category).
We filter to English tickets and sample 30 for the demo run (`N_SAMPLES` in `auto_tag.py`).

## Methodology / Approach
1. **Prompt/label engineering, no fine-tuning**: use `facebook/bart-large-mnli` via Hugging
   Face's `zero-shot-classification` pipeline, which frames tagging as natural language
   inference between the ticket text and each candidate label.
2. Candidate labels are the 10 real ticket queues (Technical Support, Billing and Payments,
   Returns and Exchanges, etc.) — a purely prompt-driven, zero-shot approach requiring no
   labeled training data.
3. **Zero-shot vs. few-shot-style prompting comparison**: two hypothesis templates are compared —
   a plain template (`"This support ticket is about {}."`) vs. a few-shot-style template that
   enriches the hypothesis with example categories (billing disputes, login issues, refund
   requests, outage reports) to give the NLI model more context.
4. `multi_label=True` lets each ticket score independently against every label, so the top 3
   scores are reported per ticket rather than a single forced choice.
5. Both approaches are evaluated by top-1 accuracy against the ticket's real `queue` label.

## How to Run

```bash
pip install -r requirements.txt
python auto_tag.py
```

## Results
- `tagging_results.csv` — per-ticket top-3 tags (with scores) from both prompting strategies,
  alongside the ground-truth queue.
- `eval_results.json` — summary top-1 accuracy for zero-shot vs. few-shot-style prompting.

## Key Results / Insights
- Zero-shot NLI-based classification is a strong baseline for ticket tagging with **zero
  labeled training data** — useful when categories evolve faster than a labeled dataset can
  keep up.
- The few-shot-style prompt (adding example scenarios to the hypothesis template) shifts the
  score distribution and can improve top-1 accuracy on ambiguous tickets, though it does not
  uniformly beat the plain template — a reminder that hypothesis wording matters for NLI-based
  zero-shot classification.
- Reporting top 3 tags (not just top 1) better matches how support tickets are triaged in
  practice, since tickets often span more than one category (e.g. "Billing" + "Technical
  Support" for a failed payment retry).

## Skills Gained
- Prompt engineering
- LLM-based text classification
- Zero-shot and few-shot learning
- Multi-class prediction and ranking
