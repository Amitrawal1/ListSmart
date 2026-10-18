from app.pricing import (
    compute_stats, margin_pct, mock_competitors, parse_price, remove_outliers_iqr,
    round_to_ending_9, round_up_to_ending_9, suggest_price,
)


def test_parse_price_formats():
    assert parse_price("₹1,299.00") == 1299.0
    assert parse_price("Rs. 499") == 499.0
    assert parse_price(349) == 349.0
    assert parse_price("free") is None
    assert parse_price(None) is None
    assert parse_price(0) is None


def test_round_to_ending_9():
    assert round_to_ending_9(474.05) == 479
    assert round_to_ending_9(463) == 459
    assert round_to_ending_9(500) == 499
    assert round_to_ending_9(3) == 9
    assert round_up_to_ending_9(461) == 469
    assert round_up_to_ending_9(469) == 469


def test_iqr_drops_outlier():
    prices = [400, 420, 450, 480, 500, 510, 5000]
    assert 5000 not in remove_outliers_iqr(prices)
    assert len(remove_outliers_iqr([1, 1000])) == 2  # too few to judge


def test_stats():
    s = compute_stats([100, 200, 300, 400, 500])
    assert s == {"count": 5, "min": 100, "median": 300, "max": 500, "p25": 200, "p75": 400}
    assert compute_stats([])["median"] is None


def test_suggest_undercuts_median():
    stats = compute_stats([299, 399, 499, 599, 899])
    price, reasons = suggest_price(stats)
    assert price == 479  # 499 * 0.95 = 474 -> 479
    assert reasons


def test_suggest_cost_floor_wins():
    stats = compute_stats([299, 399, 499, 599, 899])
    price, _ = suggest_price(stats, cost_price=500)
    assert price == 629  # 500 * 1.25 = 625 -> 629
    assert margin_pct(price, 500) == round((629 - 500) / 629 * 100, 1)


def test_suggest_clamped_to_iqr():
    # Tight market: median*0.95 falls under p25, so it should be raised into range
    stats = compute_stats([1000, 1000, 1000, 1000, 1000])
    price, _ = suggest_price(stats)
    assert 1000 <= price <= 1000


def test_no_competitors():
    assert suggest_price(compute_stats([]))[0] is None
    assert suggest_price(compute_stats([]), cost_price=100)[0] == 129


def test_mock_competitors_deterministic():
    assert mock_competitors("maroon kurti") == mock_competitors("maroon kurti")
    assert len(mock_competitors("x")) == 10
