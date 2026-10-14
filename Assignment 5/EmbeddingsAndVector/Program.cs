
using Google.GenAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.VectorData;
using CommunityToolkit.VectorData.PgVector;

const int EmbeddingDimensions = 768;

var configuration = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

var apiKey = configuration["Gemini:ApiKey"];

var connectionString = configuration["ConnectionStrings:Postgres"];

if (string.IsNullOrWhiteSpace(apiKey))
{
    throw new InvalidOperationException(
        "Gemini API key is missing. Configure User Secrets.");
}

if (string.IsNullOrWhiteSpace(connectionString))
{
    throw new InvalidOperationException(
        "Neon connection string is missing. Configure User Secrets.");
}

var geminiClient = new Client(apiKey: apiKey);

IEmbeddingGenerator<string, Embedding<float>> embedder =
    geminiClient.AsIEmbeddingGenerator(
        "gemini-embedding-001",
        EmbeddingDimensions);

var store = new PostgresVectorStore(
    connectionString,
    new()
    {
        EmbeddingGenerator = embedder
    });

var collection = store.GetCollection<string, Note>("notes");

await collection.EnsureCollectionExistsAsync();

var notes = new List<Note>
{
    new()
    {
        Id = "1",
        Topic = "API",
        Text = "Our API returns HTTP 429 when a client sends more than 100 requests per minute."
    },
    new()
    {
        Id = "2",
        Topic = "API",
        Text = "Clients must include a valid JWT access token in the Authorization header."
    },
    new()
    {
        Id = "3",
        Topic = "API",
        Text = "The health endpoint returns HTTP 200 when the service is running correctly."
    },
    new()
    {
        Id = "4",
        Topic = "Deployments",
        Text = "Production deployments happen every Tuesday and Thursday after 6 PM IST."
    },
    new()
    {
        Id = "5",
        Topic = "Deployments",
        Text = "Before releasing a new version, run automated tests and obtain approval."
    },
    new()
    {
        Id = "6",
        Topic = "Deployments",
        Text = "If a release causes errors, roll back to the last stable application version."
    },
    new()
    {
        Id = "7",
        Topic = "Onboarding",
        Text = "New developers must install the .NET SDK and configure access to the source repository."
    },
    new()
    {
        Id = "8",
        Topic = "Onboarding",
        Text = "Store development secrets in User Secrets instead of committing credentials to Git."
    },
    new()
    {
        Id = "9",
        Topic = "Incidents",
        Text = "When production is unavailable, page the on-call engineer and share the incident details."
    },
    new()
    {
        Id = "10",
        Topic = "Incidents",
        Text = "After resolving an incident, document the root cause, impact, and corrective actions."
    }
};

foreach (var note in notes)
{
    var response = await embedder.GenerateAsync(new[] { note.Text });
    note.Embedding = response[0].Vector;
}

await collection.UpsertAsync(notes);
Console.WriteLine($"Stored {notes.Count} notes in 'notes'");

Console.WriteLine($"Stored {notes.Count} notes in 'notes'");

var queries = new[]
{
    "How do I avoid sending too many API requests?",
    "Everything is down. Who should I contact?",
    "What should happen if the latest release breaks production?"
};

foreach (var query in queries)
{
    Console.WriteLine($"\nSearch: {query}");

    await foreach (var result in collection.SearchAsync(
        query,
        top: 3))
    {
        Console.WriteLine(
            $"{result.Score:F3} " +
            $"[{result.Record.Topic}] {result.Record.Text}");
    }
}

Console.WriteLine("\nFiltered search: Deployments");

var options = new VectorSearchOptions<Note>
{
    Filter = n => n.Topic == "Deployments"
};

await foreach (var result in collection.SearchAsync(
    "What should happen if a release fails?",
    top: 3,
    options))
{
    Console.WriteLine(
        $"{result.Score:F3} " +
        $"[{result.Record.Topic}] {result.Record.Text}");
}

Console.WriteLine("\nUnrelated search:");

var foundMatch = false;

await foreach (var result in collection.SearchAsync(
    "Where can I find the best biryani in Hyderabad?",
    top: 3))
{
    if (result.Score < 0.30)
    {
        continue;
    }

    foundMatch = true;

    Console.WriteLine(
        $"{result.Score:F3} " +
        $"[{result.Record.Topic}] {result.Record.Text}");
}

if (!foundMatch)
{
    Console.WriteLine("No good match found.");
}

public sealed class Note
{
    [VectorStoreKey]
    public string Id { get; set; } = string.Empty;

    [VectorStoreData]
    public string Topic { get; set; } = string.Empty;

    [VectorStoreData]
    public string Text { get; set; } = string.Empty;

    [VectorStoreVector(768, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float>? Embedding { get; set; }
}