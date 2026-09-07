"""Pick the high-ID tokens worth keeping with our spare budget.

Sources, in priority order:
  1. every high-ID token actually used by egoproactive val700 (26)
  2. high-ID tokens from a procedural-domain word list (cooking / DIY / crafts / tools /
     household) — the hidden test is the same task family, so these are the likely rare words
Capped so the final vocab stays comfortably under 2B.
"""
import json, sys
from transformers import AutoProcessor

KEEP_LOW = 143000
ADDED_START = 248044
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 600      # extra rows we allow ourselves

DOMAIN = """
sauté sautéed saute braise braised blanch blanched julienne dice diced mince minced
carton chia muesli ramen kebab quinoa couscous risotto tofu tempeh miso tahini hummus
paprika turmeric cumin coriander cardamom cinnamon nutmeg oregano thyme rosemary basil
parsley cilantro chives shallot scallion leek zucchini courgette aubergine eggplant
broccoli cauliflower asparagus artichoke avocado mango papaya pomegranate raspberry
blueberry cranberry apricot nectarine cantaloupe honeydew rhubarb parsnip turnip
casserole skillet saucepan colander sieve whisk spatula ladle tongs peeler grater
mandoline ramekin baking parchment skewer trivet cleaver paring serrated
screwdriver ratchet socket wrench pliers chisel mallet sander sandpaper grommet
dowel bracket anchor stud joist grout caulk sealant primer varnish lacquer shellac
plywood laminate veneer drywall spackle putty solder flux ferrule washer bushing
crochet knitting embroidery quilting applique bobbin thimble pinking selvage
decoupage papier mache origami quilling calligraphy watercolour gouache
trowel secateurs pruner mulch compost fertiliser fertilizer perlite vermiculite
hydrangea succulent terracotta planter trellis
detergent squeegee microfibre microfiber vacuum upholstery grouting descale limescale
"""


def main():
    proc = AutoProcessor.from_pretrained("/models/proactive")
    tok = proc.tokenizer
    observed = set(json.load(open("/work/rare_ids.json")))          # from count_rare.py
    extras = list(observed)
    seen = set(extras)
    for w in DOMAIN.split():
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            for t in tok.encode(form, add_special_tokens=False):
                if KEEP_LOW <= t < ADDED_START and t not in seen:
                    seen.add(t)
                    extras.append(t)
    extras = sorted(extras)[:CAP]
    n_final = KEEP_LOW + len(extras) + 33
    params = 2.2132e9 - (248320 - n_final) * 2048
    print(f"observed(val700): {len(observed)}   domain-added: {len(extras)-len(observed)}")
    print(f"extras kept: {len(extras)}  -> final vocab {n_final:,}  params ~{params/1e9:.4f} B "
          f"({'UNDER' if params < 2e9 else 'OVER'} 2B, margin {(2e9-params)/1e6:.1f} M)")
    json.dump(extras, open("/work/extra_ids.json", "w"))


main()
