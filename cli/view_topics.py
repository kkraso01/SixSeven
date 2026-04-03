"""Helper script to view and select conspiracy topics for experiments."""

from debate.core.topics import (
    CONSPIRACY_TOPICS,
    get_topics_by_category,
    list_categories,
)


def print_all_topics():
    """Print all available topics organized by category."""
    print("\n" + "=" * 80)
    print("CONSPIRACY THEORY DEBATE TOPICS")
    print("=" * 80 + "\n")

    categories = list_categories()

    for category in categories:
        print(f"\n {category.upper()}")
        print("-" * 80)
        topics = get_topics_by_category(category)
        for topic in topics:
            print(f"\n   {topic.id}")
            print(f"     Topic: {topic.topic}")
            print(f"     Motion: {topic.motion}")

    print(f"\n{'=' * 80}")
    print(f"Total Topics: {len(CONSPIRACY_TOPICS)}")
    print(f"Categories: {len(categories)}")
    print("=" * 80 + "\n")


def print_category_summary():
    """Print summary of topics by category."""
    print("\n" + "=" * 80)
    print("TOPICS BY CATEGORY")
    print("=" * 80 + "\n")

    categories = list_categories()
    for category in categories:
        topics = get_topics_by_category(category)
        print(f"{category:15s}: {len(topics):2d} topics")

    print(f"\n{'=' * 80}")
    print(f"Total: {len(CONSPIRACY_TOPICS)} topics")
    print("=" * 80 + "\n")


def print_topic_ids():
    """Print just the topic IDs for easy copy-paste."""
    print("\n" + "=" * 80)
    print("TOPIC IDs (for batch_runner.py)")
    print("=" * 80 + "\n")

    for topic in CONSPIRACY_TOPICS:
        print(f"  '{topic.id}',")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "summary":
            print_category_summary()
        elif command == "ids":
            print_topic_ids()
        elif command == "all":
            print_all_topics()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python view_topics.py [summary|ids|all]")
    else:
        print_category_summary()
