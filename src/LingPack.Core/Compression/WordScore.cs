namespace LingPack.Core.Compression;

/// <summary>One word's preserve-probability and whether it survived compression.</summary>
public sealed record WordScore(string Text, float PreserveProbability, bool Kept);
