# Forecasting procedure

Distilled from two batches of runs on this task family and revised once against
what the second batch measured. Every rule here is about how to use the tools and
how to state an answer.

## Numeric questions: the anchor is the answer

This is where most points are lost, and the cause is measurable. Across a batch
of numeric questions, simply reporting the most recent published value that was
available at the information cutoff scored **0.72**, while the answers actually
produced scored **0.44**. Two thirds of the time the plain latest value would
have been the better answer. The failures were not near misses: four in ten were
off by more than the scoring rule's entire tolerance, and the worst were wrong by
four orders of magnitude while the correct value sat in the evidence already
gathered.

So:

1. **Find the series, not a number.** Identify the exact quantity the question
   names — the same publisher, the same instrument, the same field. `open price`
   is not `closing price`; `cash selling rate` is not `remittance buying rate`;
   a forecast endpoint is not an archive endpoint. If the source offers several
   fields, name which one the question asks for before reading any value.

2. **Take the most recent published value at or before your information cutoff.
   That value is your answer.** Write it down before you reason about the future.

3. **Copy the unit and scale from the source, unchanged.** Tonnes are not
   kilograms, `万` is not `1`, an index level is not an index change, a rate per
   100 units is not a rate per unit. Errors of four orders of magnitude come from
   this step and from nowhere else. After writing the number, compare its
   magnitude with the anchor you just recorded: if they differ by more than a
   factor of two, you have converted something you should not have.

4. **Move off the anchor only for a reason you can name and date.** A scheduled
   release, a known policy change, a contract roll, a seasonal reset. "The trend
   is upward", "it has been rising", "momentum suggests" are not reasons — over a
   one-week horizon they are noise, and the scoring rule measures your error
   against how much the series normally moves, so an unjustified adjustment
   costs more than it can win. When you do adjust, adjust by less than one
   typical between-publication move.

5. **If you cannot find the series, say the closest value you did find** rather
   than constructing a number by inference. A stale anchor beats a derived guess.

Report the bare number in the source's own unit. No thousands separators, no
currency symbols, no unit words, unless the question's own options carry them.

## Searching

A search engine matches words that appear **on** a page, not the page's address.
Do not put a URL, a path, or a query string into a query — not after `site:` and
not as bare keywords. Name the publisher as a word instead.

When a query returns nothing, that means the phrasing found nothing, not that the
evidence is unavailable. Change the angle: drop the rarest term, use the
publisher's vocabulary, or search for the quantity rather than the event.
Re-issuing the same intent with the punctuation moved returns the same nothing.

## Fetching

Fetch a data endpoint directly by constructing its URL. Do not search for it.

**Never fetch the same URL twice.** If a page returned something, you already
have it; if it returned nothing useful, it will return nothing useful again.
Repeating one address four or five times is the most common way of exhausting the
tool budget without gaining a single fact. If a source did not yield the value,
change the source, not the attempt count.

A live-quote or "current value" endpoint answers with today's number. On a
backtest that number is outside your information boundary and will be withheld.
Look for the dated or historical form of the same source instead.

## Multiple choice

Read the resolution criteria before comparing options: the threshold, the
direction, and the deadline decide the answer more often than the topic does.
When two options differ by a single qualifier, settle that qualifier directly
rather than picking the option that reads more completely.

Answer with one option written exactly as the question lists it.

## Budget and confidence

Spend tool calls on distinct angles, not repeats. Reserve enough of the budget to
write the answer: an answer from partial evidence scores far better than none.

State a probability that matches how firmly the evidence pins the answer. Note
that changing a well-supported answer costs more than adding a weakly-supported
one gains — when the evidence for your current answer is solid, do not overturn
it on a late, weaker signal.
