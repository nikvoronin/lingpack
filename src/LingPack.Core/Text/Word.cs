namespace LingPack.Core.Text;

/// <summary>
/// A single word from the input text, together with the exact whitespace that preceded it,
/// so the original string can be reconstructed exactly when every word is kept.
/// </summary>
public readonly record struct Word(string Text, string LeadingWhitespace, int StartIndex, int Length);
