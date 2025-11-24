from agent.tools import filter_materials


def test_filter_materials_applies_all_filters(patch_materials):
    """Ensure structured filtering respects category, price, eco, and VOC."""
    results = filter_materials(
        category="flooring",
        max_price=80,
        min_eco=0.8,
        voc=1,  # low VOC
    )

    assert results, "Expected at least one flooring material to be returned"
    assert results[0]["name"] == "Eco Friendly Bamboo Flooring"

    for item in results:
        assert "floor" in item["category"].lower()
        assert item["price_per_m2"] <= 80
        assert item["eco_score"] >= 0.8
        assert item["voc_level_num"] <= 1

