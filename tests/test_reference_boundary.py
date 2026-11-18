import asyncio

from pathsnap.api import MatchRequest, create_app


def test_private_reference_changes_metrics_without_changing_matching(tmp_path):
    trip_id = '00001234'
    inputs = tmp_path / 'inputs'
    trip = inputs / trip_id
    trip.mkdir(parents=True)
    references = tmp_path / 'references'
    references.mkdir()
    (trip / f'{trip_id}.track').write_text('12.0001 45 0\n12.0002 45 1\n12.0003 45 2\n')
    (trip / f'{trip_id}.nodes').write_text('12 45\n12.001 45\n')
    (trip / f'{trip_id}.arcs').write_text('0 1\n')
    reference = references / f'{trip_id}.route'
    reference.write_text('0\n')
    api = create_app(evaluator_root=inputs, reference_root=references)
    match = next(route.endpoint for route in api.routes if getattr(route, 'path', None) == '/api/match')
    request = MatchRequest(partition='evaluator_test', trip_id=trip_id)

    async def verify():
        async with api.router.lifespan_context(api):
            scored = match(request)
            reference.unlink()
            unscored = match(request)
        assert scored['reference_metrics_available']
        assert not unscored['reference_metrics_available']
        assert scored['map'] == unscored['map']
        assert scored['metrics']['hmm']['proxy_observations'] > 0
        assert 'proxy_accuracy' not in unscored['metrics']['hmm']

    asyncio.run(verify())
