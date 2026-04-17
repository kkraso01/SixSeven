"""Quick test to verify search functionality."""

from debate.simulator.providers.search import search_web

# Test search
print("Testing DuckDuckGo search...")
print("=" * 60)

result = search_web("site:reddit.com/r/conspiracy covid vaccine", max_results=3)

if result.success:
    print(" Search successful!")
    print(f"Query: {result.query}")
    print(f"Results found: {len(result.results)}")
    print("\nFirst result:")
    if result.results:
        print(f"  Title: {result.results[0].title}")
        print(f"  URL: {result.results[0].href}")
        print(f"  Snippet: {result.results[0].body[:100]}...")
else:
    print(f" Search failed: {result.error}")

print("=" * 60)
