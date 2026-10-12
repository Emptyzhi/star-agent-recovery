"""
"""
import os
import re
from urllib.parse import unquote
import json
import random
from collections import defaultdict
from ast import literal_eval
from decimal import Decimal

import cleantext
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from flask import render_template_string
from rich import print
from pyserini.search.lucene import LuceneSearcher

from web_agent_site.utils import (
    BASE_DIR,
    BENCHMARK_ROOT,
    DEFAULT_FILE_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH
)

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

SEARCH_RETURN_N = 50
PRODUCT_WINDOW = 10
TOP_K_ATTR = 10

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            reward=kwargs['reward'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            reward_info=kwargs.get('reward_info'),
            goal_attrs=kwargs.get('goal_attrs'),
            purchased_attrs=kwargs.get('purchased_attrs'),
            goal=kwargs.get('goal'),
            mturk_code=kwargs.get('mturk_code'),
            query=kwargs.get('query'),
            category=kwargs.get('category'),
            product_category=kwargs.get('product_category'),
        )
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text')
        )
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs']
        )
    else:
        raise ValueError('Action name not recognized.')
    return html


def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template


def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    pattern = re.compile(r'(.+)\[(.+)\]')
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        # The client protects slashes from Flask route splitting with one
        # transport encoding layer. Remove only that layer here before the
        # native search parser sees the query.
        keywords = unquote(string)
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var


def get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins=None,
    ):
    if keywords[0] == '<r>':
        top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
    elif keywords[0] == '<a>':
        attribute = ' '.join(keywords[1:]).strip()
        asins = attribute_to_asins[attribute]
        top_n_products = [p for p in all_products if p['asin'] in asins]
    elif keywords[0] == '<c>':
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p['category'] == category]
    elif keywords[0] == '<q>':
        query = ' '.join(keywords[1:]).strip()
        top_n_products = [p for p in all_products if p['query'] == query]
    else:
        keywords = ' '.join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        docs = [search_engine.doc(hit.docid) for hit in hits]
        top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    return top_n_products


def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


def init_search_engine(num_products=None):
    if num_products == 100:
        indexes = 'indexes_100'
    elif num_products == 1000:
        indexes = 'indexes_1k'
    elif num_products == 100000:
        indexes = 'indexes_100k'
    elif num_products is None:
        indexes = 'indexes'
    else:
        raise NotImplementedError(f'num_products being {num_products} is not supported yet.')
    search_engine = LuceneSearcher(os.path.join(BENCHMARK_ROOT, f'search_engine/{indexes}'))
    return search_engine


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products


def _iter_json_array(filepath):
    """Stream a top-level JSON array without retaining the raw catalog."""
    decoder = json.JSONDecoder()
    with open(filepath, encoding='utf-8') as stream:
        buffer = ''
        position = 0
        eof = False

        def refill():
            nonlocal buffer, eof
            chunk = stream.read(8 * 1024 * 1024)
            if chunk:
                buffer += chunk
                return True
            eof = True
            return False

        refill()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer) or buffer[position] != '[':
            raise ValueError('WebShop product file must contain a JSON array')
        position += 1
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if eof or not refill():
                    raise ValueError('unexpected end of WebShop product array')
            if buffer[position] == ']':
                return
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof or not refill():
                        raise
                    continue
                yield item
                position = end
                break
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if eof or not refill():
                    raise ValueError('unexpected end after WebShop product')
            if buffer[position] == ',':
                position += 1
            elif buffer[position] == ']':
                return
            else:
                raise ValueError('invalid WebShop product array separator')
            if position >= 8 * 1024 * 1024:
                buffer = buffer[position:]
                position = 0


def load_products(filepath, num_products=None, human_goals=True):
    all_reviews = {}
    all_ratings = {}
    if human_goals:
        with open(HUMAN_ATTR_PATH, encoding='utf-8') as stream:
            human_attributes = json.load(stream)
    with open(DEFAULT_ATTR_PATH, encoding='utf-8') as stream:
        attributes = json.load(stream)
    with open(HUMAN_ATTR_PATH, encoding='utf-8') as stream:
        human_attributes = json.load(stream)
    print('Attributes loaded; streaming products.')

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    iterator = _iter_json_array(filepath)
    for raw_index, p in enumerate(tqdm(iterator, total=num_products)):
        if num_products is not None and raw_index >= num_products:
            break
        asin = p['asin']
        if asin == 'nan' or len(asin) > 10 or asin in asins:
            continue
        asins.add(asin)
        for key in ('product_information', 'brand', 'brand_url', 'list_price',
                    'availability_quantity', 'availability_status', 'total_reviews',
                    'total_answered_questions', 'seller_id', 'seller_name',
                    'fulfilled_by_amazon', 'fast_track_message', 'aplus_present',
                    'small_description_old'):
            p.pop(key, None)
        p['category'] = p['category']
        p['query'] = p['query']
        p['product_category'] = p['product_category']
        p['Title'] = p['name']
        p['Description'] = p['full_description']
        p['Reviews'] = all_reviews.get(asin, [])
        p['Rating'] = all_ratings.get(asin, 'N.A.')
        for review in p['Reviews']:
            if 'score' not in review:
                review['score'] = review.pop('stars')
            if 'review' not in review:
                review['body'] = ''
            else:
                review['body'] = review.pop('review')
        p['BulletPoints'] = p['small_description'] if isinstance(p['small_description'], list) else [p['small_description']]
        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing, price_tag = [100.0], '$100.0'
        else:
            pricing = [float(Decimal(re.sub(r'[^\d.]', '', price))) for price in pricing.split('$')[1:]]
            if len(pricing) == 1:
                price_tag = f'${pricing[0]}'
            else:
                price_tag = f'${pricing[0]} to ${pricing[1]}'
                pricing = pricing[:2]
        p['pricing'], p['Price'] = pricing, price_tag
        options, option_to_image = {}, {}
        for option_name, option_contents in (p['customization_options'] or {}).items():
            if option_contents is None:
                continue
            values = []
            for option_content in option_contents:
                value = option_content['value'].strip().replace('/', ' | ').lower()
                values.append(value)
                option_to_image[value] = option_content.get('image')
            options[option_name.lower()] = values
        p['options'], p['option_to_image'] = options, option_to_image
        p['Attributes'] = attributes.get(asin, {}).get('attributes', ['DUMMY_ATTR'])
        if human_goals:
            if asin in human_attributes:
                p['instructions'] = human_attributes[asin]
        else:
            p['instruction_text'] = attributes[asin].get('instruction')
            p['instruction_attributes'] = attributes[asin].get('instruction_attributes')
        p['MainImage'] = p['images'][0]
        p['query'] = p['query'].lower().strip()
        all_products.append(p)
        for attribute in p['Attributes']:
            attribute_to_asins[attribute].add(asin)
    print('Products loaded.')
    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices, attribute_to_asins
