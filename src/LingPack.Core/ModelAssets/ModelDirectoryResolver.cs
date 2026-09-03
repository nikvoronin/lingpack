namespace LingPack.Core.ModelAssets;

/// <summary>
/// Resolves the directory holding the exported ONNX model + tokenizer assets, and validates the
/// expected files are present. Precedence: explicit flag &gt; <c>LINGPACK_MODEL_DIR</c> env var &gt;
/// default under %LOCALAPPDATA%.
/// </summary>
public static class ModelDirectoryResolver
{
    public const string EnvironmentVariableName = "LINGPACK_MODEL_DIR";

    public static string DefaultModelDirectory =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "lingpack", "models", "llmlingua-2-xlm-roberta-large");

    public static string Resolve(string? modelDirFlag)
        => modelDirFlag
           ?? Environment.GetEnvironmentVariable(EnvironmentVariableName)
           ?? DefaultModelDirectory;

    public static ModelAssetPaths ResolveAndValidate(string? modelDirFlag)
    {
        var dir = Resolve(modelDirFlag);

        var missing = ModelAssetPaths.RequiredFileNames
            .Where(name => !File.Exists(Path.Combine(dir, name)))
            .ToArray();

        if (missing.Length > 0)
        {
            throw new FileNotFoundException(
                $"Model directory '{dir}' is missing required file(s): {string.Join(", ", missing)}. " +
                "Run scripts/export-model.py (or scripts/export-model.ps1) once to export the model, " +
                $"or point --model-dir / {EnvironmentVariableName} at a directory that already contains it.");
        }

        return new ModelAssetPaths(
            OnnxModelPath: Path.Combine(dir, "model.onnx"),
            SentencePieceModelPath: Path.Combine(dir, "sentencepiece.bpe.model"),
            TokenizerJsonPath: Path.Combine(dir, "tokenizer.json"));
    }
}
