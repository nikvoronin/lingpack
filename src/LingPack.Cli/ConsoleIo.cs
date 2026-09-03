namespace LingPack.Cli;

internal static class ConsoleIo
{
    public static async Task<string> ReadInputAsync(FileInfo? inputFile, CancellationToken cancellationToken)
        => inputFile is not null
            ? await File.ReadAllTextAsync(inputFile.FullName, cancellationToken)
            : await Console.In.ReadToEndAsync(cancellationToken);

    public static async Task WriteOutputAsync(FileInfo? outputFile, string text, CancellationToken cancellationToken)
    {
        if (outputFile is not null)
        {
            await File.WriteAllTextAsync(outputFile.FullName, text, cancellationToken);
        }
        else
        {
            await Console.Out.WriteAsync(text);
            await Console.Out.WriteLineAsync();
        }
    }
}
