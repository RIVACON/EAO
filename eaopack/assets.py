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

### import other basic classes as well (for convenience)
from eaopack.basic_classes import Timegrid, Unit, Node