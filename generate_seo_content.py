async def generate_seo_content(description):
    # Generate SEO title from the first sentence
    seo_title = description.split(".")[0]
    print(f"\nGenerated SEO Title:\n{seo_title}")

    # Generate SEO meta description from the first two sentences
    seo_meta_description = " ".join(description.split(".")[:2])
    print(f"\nGenerated SEO Meta Description:\n{seo_meta_description}")

    # Generate tags by splitting the description into words and picking the first few unique words
    tags = list(set(description.split()))[:10]
    print(f"\nGenerated Product Tags:\n{tags}")

    # Generate focus keywords by picking the first two unique words
    focus_keywords = " ".join(tags[:2])
    print(f"\nGenerated Focus Keywords:\n{focus_keywords}")

    return seo_title, seo_meta_description, tags, focus_keywords
