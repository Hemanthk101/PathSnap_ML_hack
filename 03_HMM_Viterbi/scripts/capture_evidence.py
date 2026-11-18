"""Verify dashboard interactions in Chromium and capture report screenshots."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def load_student_demo(page, url):
    def match_response(response):
        return '/api/match' in response.url and response.request.method == 'POST'

    with page.expect_response(match_response, timeout=90000) as initial:
        page.goto(url, wait_until='domcontentloaded')
    result = initial.value.json()
    if result['partition'] != 'student_test':
        with page.expect_response(match_response, timeout=90000) as selected:
            page.get_by_role('combobox', name='Trip partition').select_option('student_test')
        result = selected.value.json()
    assert result['partition'] == 'student_test'
    expect(page.locator('#map-state')).to_be_hidden(timeout=90000)
    return result


def dashboard(browser, url, folder):
    page = browser.new_page(viewport={'width': 1440, 'height': 920}, accept_downloads=True)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    first = load_student_demo(page, url)
    expect(page.locator('.trip-row')).to_have_count(10)
    expect(page.locator('#m-observations')).to_have_text(f'{first["metrics"]["hmm"]["observations"]:,}')
    expect(page.locator('#api-status')).to_contain_text('API connected')
    assert page.evaluate('document.documentElement.scrollWidth === innerWidth')
    page.screenshot(path=folder / 'dashboard.png', full_page=True, animations='disabled')

    page.get_by_role('button', name='Classifier', exact=True).click()
    expect(page.get_by_role('button', name='Classifier', exact=True)).to_have_attribute('aria-pressed', 'true')
    page.get_by_role('button', name='Raw GPS').click()
    expect(page.get_by_role('button', name='Raw GPS')).to_have_attribute('aria-pressed', 'false')
    page.get_by_role('button', name='Raw GPS').click()
    page.get_by_role('button', name='Zoom in').click()
    map_box = page.locator('#map').bounding_box()
    page.mouse.move(map_box['x'] + 330, map_box['y'] + 150)
    page.mouse.down()
    page.mouse.move(map_box['x'] + 356, map_box['y'] + 185, steps=5)
    page.mouse.up()
    page.get_by_role('button', name='Fit route').click()

    page.get_by_role('slider').fill('58')
    expect(page.locator('#obs-counter')).to_contain_text('59 /')
    page.get_by_role('button', name='First observation').click()
    expect(page.locator('#obs-counter')).to_contain_text('1 /')
    page.get_by_role('combobox', name='Playback speed').select_option('40')
    page.get_by_role('button', name='Play trace').click()
    page.wait_for_function("Number(document.querySelector('#timeline').value) > 0")
    page.get_by_role('button', name='Pause trace').click()

    with page.expect_download() as downloaded:
        page.get_by_role('button', name='Export GeoJSON').click()
    feature_collection = json.loads(downloaded.value.path().read_text(encoding='utf-8'))
    assert feature_collection['type'] == 'FeatureCollection' and feature_collection['features']

    page.get_by_role('searchbox').fill('00000052')
    expect(page.locator('.trip-row')).to_have_count(1)
    with page.expect_response(lambda response: '/api/match' in response.url, timeout=90000) as selected:
        page.locator('.trip-row').click()
    assert selected.value.json()['trip_id'] == '00000052'
    expect(page.locator('#map-state')).to_be_hidden(timeout=90000)
    expect(page.locator('#active-trip')).to_have_text('00000052')
    page.get_by_role('searchbox').fill('xyz')
    expect(page.get_by_text('No trip IDs match your search.')).to_be_visible()

    page.get_by_role('button', name='Experiments', exact=True).click()
    expect(page.get_by_role('heading', name='Model comparison')).to_be_visible(timeout=30000)
    expect(page.locator('tbody tr')).to_have_count(6)
    page.locator('.trial').first.click()
    expect(page.locator('#trial-caption')).to_contain_text('Trial 1')
    page.screenshot(path=folder / 'experiments.png', full_page=True, animations='disabled')
    assert not errors, errors

    mobile = browser.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
    load_student_demo(mobile, url)
    assert mobile.evaluate('document.documentElement.scrollWidth === innerWidth')
    mobile.screenshot(path=folder / 'mobile.png', full_page=True, animations='disabled')
    mobile.close()
    page.close()


def folium(browser, html, folder):
    page = browser.new_page(viewport={'width': 1260, 'height': 680})
    page.goto(html.resolve().as_uri(), wait_until='load', timeout=60000)
    expect(page.locator('.leaflet-control-layers')).to_be_visible(timeout=25000)
    expect(page.locator('.leaflet-overlay-pane canvas')).to_have_count(1)
    page.screenshot(path=folder / 'folium.png', animations='disabled')
    page.close()


def api_response(browser, url, folder):
    page = browser.new_page(viewport={'width': 1200, 'height': 900})
    page.goto(f'{url.rstrip("/")}/docs', wait_until='load', timeout=60000)
    operation = page.locator('#operations-default-match_api_match_post')
    expect(operation).to_be_visible(timeout=30000)
    operation.locator('.opblock-summary').click()
    operation.get_by_role('button', name='Try it out').click()
    operation.locator('textarea').fill('{"trip_id":"00000052","partition":"student_test"}')
    with page.expect_response(lambda response: response.url.endswith('/api/match'), timeout=90000) as live:
        operation.get_by_role('button', name='Execute', exact=True).click()
    assert live.value.status == 200
    assert live.value.json()['trip_id'] == '00000052'
    result = operation.locator('.live-responses-table')
    expect(result).to_contain_text('200', timeout=20000)
    result.scroll_into_view_if_needed()
    page.screenshot(path=folder / 'api.png', animations='disabled')
    page.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--out', type=Path, default=Path('results/screenshots'))
    parser.add_argument('--folium', type=Path)
    parser.add_argument('--folium-only', action='store_true')
    parser.add_argument('--browser', default='chrome')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.browser or None)
        if not args.folium_only:
            dashboard(browser, args.url, args.out)
            api_response(browser, args.url, args.out)
        if args.folium:
            folium(browser, args.folium, args.out)
        browser.close()
    print('Browser interaction checks and requested screenshot captures passed.')
