import polars as pl

from src.phase7_text import normalize_review_text, prepare_eligible_reviews


def test_normalization_preserves_negation_and_punctuation():
    assert normalize_review_text("Not good!", "Very late...") == "not good! very late..."


def test_cutoff_excludes_boundary_and_future_reviews():
    reviews = pl.DataFrame(
        {
            "review_id": ["before", "boundary", "future"],
            "customer_unique_id": ["c1", "c1", "c1"],
            "review_comment_title": ["A", "B", "C"],
            "review_comment_message": ["x", "y", "z"],
            "review_creation_date": pl.Series(
                ["2018-02-28", "2018-03-01", "2018-03-02"], dtype=pl.Date
            ),
        }
    )
    eligible, _ = prepare_eligible_reviews(
        reviews,
        pl.Series("customer_unique_id", ["c1"]),
    )
    assert eligible["review_id"].to_list() == ["before"]


def test_duplicate_and_empty_reviews_are_audited():
    reviews = pl.DataFrame(
        {
            "review_id": ["r1", "r1", "r2"],
            "customer_unique_id": ["c1", "c1", "c1"],
            "review_comment_title": ["Title", "Later", None],
            "review_comment_message": ["Message", "Other", "   "],
            "review_creation_date": pl.Series(
                ["2018-01-01", "2018-01-02", "2018-01-03"], dtype=pl.Date
            ),
        }
    )
    eligible, stats = prepare_eligible_reviews(
        reviews,
        pl.Series("customer_unique_id", ["c1"]),
    )
    assert eligible["review_id"].to_list() == ["r1", "r2"]
    assert stats["duplicate_review_ids_removed"] == 1
    assert stats["empty_text_reviews"] == 1
