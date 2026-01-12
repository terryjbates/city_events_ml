def test_imports():
    import city_events_ml
    from city_events_ml.io import build_target_table, TargetSpec
    from city_events_ml.features import add_time_features
    from city_events_ml.pipelines import make_ridge_pipeline
