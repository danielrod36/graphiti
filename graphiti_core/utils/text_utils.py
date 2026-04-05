"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import re

# Maximum length for entity/community summaries
MAX_SUMMARY_CHARS = 1000


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate text at or about max_chars while respecting sentence boundaries.

    Attempts to truncate at the last complete sentence before max_chars.
    If no sentence boundary is found before max_chars, truncates at max_chars.

    Args:
        text: The text to truncate
        max_chars: Maximum number of characters

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text

    # Find all sentence boundaries (., !, ?) up to max_chars
    truncated = text[:max_chars]

    # Look for sentence boundaries: period, exclamation, or question mark followed by space or end
    sentence_pattern = r'[.!?](?:\s|$)'
    matches = list(re.finditer(sentence_pattern, truncated))

    if matches:
        # Truncate at the last sentence boundary found
        last_match = matches[-1]
        return text[: last_match.end()].rstrip()

    # No sentence boundary found, truncate at max_chars
    return truncated.rstrip()


def deduplicate_summary_sentences(text: str) -> str:
    """Remove duplicate sentences from a summary string.

    Handles both exact and near-duplicate sentences that Graphiti
    accumulates through repeated edge fact appending.

    Args:
        text: Summary text with potential duplicate lines.

    Returns:
        Text with duplicate lines removed, keeping first occurrence.
    """
    if not text:
        return text

    # Split into lines (Graphiti uses \n as separator)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) <= 1:
        return text

    seen_normalized: list[str] = []
    unique_lines: list[str] = []

    for line in lines:
        # Normalize: lowercase, collapse whitespace, strip punctuation for comparison
        normalized = " ".join(line.lower().split())
        normalized = normalized.rstrip(".!?;:,")

        # Check against all seen lines
        is_duplicate = False
        for seen in seen_normalized:
            # Exact match after normalization
            if normalized == seen:
                is_duplicate = True
                break
            # Token overlap ratio (Jaccard-like)
            tokens_new = set(normalized.split())
            tokens_seen = set(seen.split())
            if tokens_new and tokens_seen:
                overlap = len(tokens_new & tokens_seen) / min(len(tokens_new), len(tokens_seen))
                if overlap > 0.9:
                    is_duplicate = True
                    break

        if not is_duplicate:
            seen_normalized.append(normalized)
            unique_lines.append(line)

    return "\n".join(unique_lines)
