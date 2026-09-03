namespace LingPack.Core.ModelAssets;

public sealed record ModelAssetPaths(string OnnxModelPath, string SentencePieceModelPath, string TokenizerJsonPath)
{
    public static readonly string[] RequiredFileNames =
    [
        "model.onnx",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "config.json",
    ];
}
