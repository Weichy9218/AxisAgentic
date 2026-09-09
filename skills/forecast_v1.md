# Forecasting procedure

These rules come from reviewing a previous batch of runs on this task family.
Every one of them is about how to use the tools and how to state an answer.

## Writing a search query

A search engine matches words that appear **on** a page, not the page's address.
Never paste a URL, a URL path, or a query string into the query — not as a
`site:` token and not as bare keywords. `example.com/data/v1/summary?x=1` matches
nothing. Search for the words a human would read on that page: the publisher's
name, the metric's name, the period, the unit.

If you want a specific site, name it as a word (`Manifold`, `USGS`, `Eastmoney`)
rather than as `site:` with a path.

When a query returns nothing, that tells you the *phrasing* found nothing. It
does not tell you the evidence is unavailable. Change the angle, not the
punctuation: drop the rarest term, use the publisher's own vocabulary, or search
for the quantity rather than the event. Re-issuing the same intent with quotes
moved, words reordered, or a slightly different site restriction wastes a turn
and returns the same nothing. Two failed attempts at one angle is the signal to
try a different angle.

## Fetching a page

If the value you need is published by an API or a data portal, fetch the
endpoint URL directly with the page-reading tool. Do not search for the endpoint
— construct it and read it.

If two fetches of the same host fail, that host is not going to work in this
run. Move to a different source for the same quantity rather than a third
attempt.

## Numeric questions

Most numeric answers that score zero are not near misses. They are wrong by a
factor of two, ten, or more, and the cause is almost always the anchor rather
than the reasoning. Before answering, do this in order:

1. **Fix the unit and the scale.** Percent or basis points, thousands or
   millions, index level or index change, cumulative or per-period, local
   currency or USD. Write down which one the question is asking for, in the
   question's own words.
2. **Find the most recent published value of exactly that quantity**, and note
   its date. This is the anchor. A value for a similar-sounding quantity is not
   an anchor.
3. **Get two or three earlier observations** of the same series so you know how
   much it typically moves between publications. The answer should sit within a
   plausible multiple of that movement.
4. **Check the order of magnitude before answering.** If your number is not
   within a factor of two of the anchor, you need a stated reason — a known
   level shift, a different reporting basis, a seasonal reset. If you cannot
   name the reason, you have the wrong quantity or the wrong unit, and the fix
   is to go back to step 1, not to adjust the number.

Report the value in the same unit and scale the question uses. Do not add
thousands separators, currency symbols, or units unless the question's own
options carry them.

## Multiple choice

Read the resolution criteria before comparing options: the threshold, the
direction, and the deadline decide the answer more often than the topic does.
When two options differ by a single qualifier — a date, an inequality, a
cumulative-versus-point distinction — settle that qualifier directly instead of
picking the option that reads more completely.

Answer with one option written exactly as the question lists it.

## Budget and confidence

You have a bounded number of tool calls. Spend them on distinct angles, not on
repeats. When roughly a fifth of the budget remains, stop gathering and write
the answer from what you have; an answer built on partial evidence scores far
better than no answer.

State a probability that matches how well the evidence actually pins the answer.
Evidence that directly fixes the quantity supports a high number. An inference
from a related series, or a base rate with no direct observation, does not.
