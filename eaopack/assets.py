import logging

logger = logging.getLogger(__name__)


### collect all assets from other files
# for easier import and access

from eaopack.assets_basic import Asset, \
                                 Storage, \
                                 SimpleContract, \
                                 Contract, \
                                 Transport, \
                                 ExtendedTransport, \
                                 MultiCommodityContract, \
                                 ScaledAsset, \
                                 OrderBook


from eaopack.assets_plants import CHPAsset, \
                                  CHPAsset_with_min_load_costs, \
                                  Plant, \
                                  CHP_PQ_diagram


from eaopack.assets_structured import StructuredAsset, \
                                      LinkedAsset

# possibility to add non-public implementations for assets
try:
    from eaopack.assets_private import *
except ImportError:
    logger.info("No private assets found.")

### import other basic classes as well (for convenience)
from eaopack.basic_classes import Timegrid, Unit, Node