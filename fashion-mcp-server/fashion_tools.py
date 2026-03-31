from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fashion-mcp-server", host="0.0.0.0", stateless_http=True)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

CATALOG = [
    {
        "id": "SKU-001",
        "name": "Slim Fit Chino Trousers",
        "category": "Bottomwear",
        "color": "Beige",
        "tags": ["slim-fit", "chino", "casual", "office-wear", "cotton"],
        "price": 49.99,
        "gender": "Men",
    },
    {
        "id": "SKU-002",
        "name": "Floral Wrap Midi Dress",
        "category": "Dresses",
        "color": "Multicolor",
        "tags": ["floral", "wrap", "midi", "summer", "feminine", "boho"],
        "price": 74.99,
        "gender": "Women",
    },
    {
        "id": "SKU-003",
        "name": "Oversized Graphic Tee",
        "category": "Topwear",
        "color": "White",
        "tags": ["oversized", "graphic", "streetwear", "unisex", "cotton"],
        "price": 29.99,
        "gender": "Unisex",
    },
    {
        "id": "SKU-004",
        "name": "Leather Biker Jacket",
        "category": "Outerwear",
        "color": "Black",
        "tags": ["leather", "biker", "edgy", "winter", "jacket"],
        "price": 189.99,
        "gender": "Unisex",
    },
    {
        "id": "SKU-005",
        "name": "High-Waist Yoga Leggings",
        "category": "Activewear",
        "color": "Navy",
        "tags": ["high-waist", "yoga", "activewear", "stretch", "leggings"],
        "price": 39.99,
        "gender": "Women",
    },
]

TAG_TAXONOMY = {
    "fit": ["slim-fit", "oversized", "regular-fit", "relaxed", "skinny"],
    "occasion": ["casual", "office-wear", "summer", "winter", "activewear", "streetwear"],
    "style": ["boho", "feminine", "edgy", "graphic", "floral"],
    "fabric": ["cotton", "leather", "stretch"],
    "silhouette": ["midi", "wrap", "biker", "high-waist"],
}


# ---------------------------------------------------------------------------
# Tool 1 — get_product_tags
# ---------------------------------------------------------------------------

@mcp.tool()
def get_product_tags(product_id: str) -> dict:
    """
    Return the auto-generated visual tags for a product SKU.

    Args:
        product_id: The SKU identifier (e.g. 'SKU-001').

    Returns:
        A dict with the product name, category, color, and list of tags.
    """
    product = next((p for p in CATALOG if p["id"] == product_id), None)
    if product is None:
        return {"error": f"Product '{product_id}' not found in catalog."}

    return {
        "product_id": product["id"],
        "name": product["name"],
        "category": product["category"],
        "color": product["color"],
        "tags": product["tags"],
        "tag_taxonomy": {
            group: [t for t in product["tags"] if t in tags]
            for group, tags in TAG_TAXONOMY.items()
            if any(t in product["tags"] for t in tags)
        },
    }


# ---------------------------------------------------------------------------
# Tool 2 — search_catalog
# ---------------------------------------------------------------------------

@mcp.tool()
def search_catalog(
    query: str,
    gender: str = "All",
    category: str = "All",
    max_price: float = 0.0,
) -> list[dict]:
    """
    Search the fashion catalog by keyword, gender, category, or price.

    Args:
        query:     Free-text keyword matched against name, tags, and color.
        gender:    Filter by 'Men', 'Women', 'Unisex', or 'All' (default).
        category:  Filter by category string, e.g. 'Dresses', or 'All'.
        max_price: Upper price limit (0.0 = no limit).

    Returns:
        A list of matching products with id, name, category, color, tags, and price.
    """
    results = []
    query_lower = query.lower()

    for p in CATALOG:
        # keyword match
        searchable = f"{p['name']} {p['color']} {' '.join(p['tags'])}".lower()
        if query_lower and query_lower not in searchable:
            continue
        # gender filter
        if gender != "All" and p["gender"] not in (gender, "Unisex"):
            continue
        # category filter
        if category != "All" and p["category"].lower() != category.lower():
            continue
        # price filter
        if max_price > 0 and p["price"] > max_price:
            continue

        results.append({
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "color": p["color"],
            "tags": p["tags"],
            "price": p["price"],
            "gender": p["gender"],
        })

    return results if results else [{"message": "No products matched your query."}]


# ---------------------------------------------------------------------------
# Tool 3 — generate_description
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_description(product_id: str, tone: str = "professional") -> dict:
    """
    Generate a marketing product description for a catalog item.

    Args:
        product_id: The SKU identifier (e.g. 'SKU-002').
        tone:       Writing tone — 'professional', 'casual', or 'luxury'.

    Returns:
        A dict with a short tagline and a longer marketing description.
    """
    product = next((p for p in CATALOG if p["id"] == product_id), None)
    if product is None:
        return {"error": f"Product '{product_id}' not found in catalog."}

    name = product["name"]
    color = product["color"]
    tags = product["tags"]
    price = product["price"]
    category = product["category"]

    # Build tone-aware copy (mock generation — swap with LLM call in production)
    if tone == "luxury":
        tagline = f"The epitome of refined style — {name}."
        description = (
            f"Crafted for the discerning wardrobe, the {name} in {color} is a "
            f"testament to effortless sophistication. Featuring {', '.join(tags[:3])}, "
            f"this {category.lower()} piece commands attention at every turn. "
            f"Priced at ${price}, it is an investment in lasting elegance."
        )
    elif tone == "casual":
        tagline = f"Your new fave: {name} 🔥"
        description = (
            f"Obsessed with {', '.join(tags[:2])} vibes? The {name} in {color} is "
            f"exactly what your wardrobe's been missing. It's comfy, it's stylish, "
            f"and at just ${price} it's a total steal. Rock it your way!"
        )
    else:  # professional (default)
        tagline = f"Introducing the {name} — style meets function."
        description = (
            f"The {name} is a {color.lower()} {category.lower()} piece designed for "
            f"the modern wardrobe. Key attributes include {', '.join(tags[:4])}, "
            f"making it a versatile choice across multiple occasions. "
            f"Available at ${price}."
        )

    return {
        "product_id": product_id,
        "name": name,
        "tone": tone,
        "tagline": tagline,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Entry point  (streamable-http for AgentCore Runtime)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
