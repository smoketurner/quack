#!/usr/bin/env bash
# Load one coherent, public-domain supply chain dataset into a workspace:
# a table of health commodity shipments (purchase orders, vendors,
# countries, Incoterms, shipment modes, delivery dates, freight and
# insurance costs) plus documents that use the same vocabulary, so the
# same entities show up in SQL, vector search, keyword search, and the
# knowledge graph. Idempotent: files already in the workspace are skipped.
#
#   scripts/demo-data.sh [WORKSPACE] [--reset]     default workspace: logistics
#
# --reset deletes every document already in the workspace first.
#
# Sources (all US government works, public domain):
#   USAID Supply Chain Management System delivery history, 2006-2015
#     https://catalog.data.gov/dataset/supply-chain-shipment-pricing-data
#     (served from the Internet Archive; data.usaid.gov is gone)
#   CBP, Importing into the United States: A Guide for Commercial Importers
#     https://www.cbp.gov/document/publications/importing-united-states
set -euo pipefail

workspace="logistics"
reset=0
for arg in "$@"; do
  case "$arg" in
  --reset) reset=1 ;;
  *) workspace="$arg" ;;
  esac
done

cache="${QUACK_DEMO_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/quack/demo}"
quack="${QUACK_BIN:-}"
if [ -z "$quack" ]; then
  if [ -x target/debug/quack ]; then
    quack=target/debug/quack
  elif command -v quack >/dev/null 2>&1; then
    quack=quack
  else
    echo "quack binary not found; run 'cargo build --bin quack' or set QUACK_BIN" >&2
    exit 1
  fi
fi

mkdir -p "$cache"

fetch() {
  local name="$1" url="$2"
  if [ ! -s "$cache/$name" ]; then
    echo "downloading $name"
    curl -fsSL -o "$cache/$name.part" "$url"
    mv "$cache/$name.part" "$cache/$name"
  fi
}

fetch shipments-raw.csv "https://web.archive.org/web/2024id_/https://data.usaid.gov/api/views/a3rc-nmf6/rows.csv?accessType=DOWNLOAD"
fetch importing-into-the-united-states.pdf "https://www.cbp.gov/sites/default/files/documents/Importing%20into%20the%20U.S.pdf"

# Column names like "po / so #" and "freight cost (usd)" become SQL-friendly
# identifiers: po_so_number, freight_cost_usd. Only the header line changes.
if [ ! -s "$cache/shipments.csv" ]; then
  awk 'NR == 1 {
         $0 = tolower($0)
         gsub(/#/, " number")
         gsub(/[^a-z0-9,]+/, "_")
         gsub(/_,/, ",")
         gsub(/,_/, ",")
         sub(/^_/, "")
         sub(/_$/, "")
       }
       { print }' "$cache/shipments-raw.csv" >"$cache/shipments.csv"
fi

cat >"$cache/shipments-data-dictionary.md" <<'DOC'
# Shipments table: data dictionary

The `shipments` table is the USAID Supply Chain Management System (SCMS) delivery history
for antiretroviral drugs (ARVs) and HIV test kits sent to partner countries between 2006
and 2015. One row is one line item on one shipment. Fields:

- `id`: row identifier.
- `project_code`: the SCMS project the line item belongs to, such as `100-CI-T01`.
- `pq_number`: price quote (PQ) reference; `Pre-PQ Process` when no quote step existed.
- `po_so_number`: purchase order (PO) or sales order (SO) number, such as `SCMS-4`.
- `asn_dn_number`: advance shipment notice (ASN) or delivery note (DN) number, such as `ASN-8`.
- `country`: destination country.
- `managed_by`: the office that managed the order, for example `PMO - US`.
- `fulfill_via`: `Direct Drop` when the vendor shipped straight to the country, or
  `From RDC` when the goods came from a regional distribution center.
- `vendor_inco_term`: the Incoterm agreed with the vendor (EXW, FCA, CIP, DDP, DDU, CIF).
  See the Incoterms document. `N/A - From RDC` for regional distribution center stock.
- `shipment_mode`: `Air`, `Ocean`, `Truck`, or `Air Charter`.
- `pq_first_sent_to_client_date`, `po_sent_to_vendor_date`, `scheduled_delivery_date`,
  `delivered_to_client_date`, `delivery_recorded_date`: milestones as text in
  day-month-year form such as `2-Jun-06`; some hold `Date Not Captured` or
  `Pre-PQ Process`. Parse with `try_strptime(col, '%d-%b-%y')`.
- `product_group`: `ARV` (antiretroviral drugs), `HRDT` (HIV rapid diagnostic tests), `ANTM`
  (antimalarials), `ACT`, `MRDT`.
- `sub_classification`: for ARVs, `Adult`, `Pediatric`; for tests, `HIV test`,
  `HIV test - Ancillary`.
- `vendor`: the supplier the order was placed with.
- `item_description`, `molecule_test_type`, `brand`, `dosage`, `dosage_form`: what was
  shipped. `molecule_test_type` names the active ingredients, such as
  `Lamivudine/Nevirapine/Zidovudine`.
- `unit_of_measure_per_pack`: units in one pack (tablets, tests, millilitres).
- `line_item_quantity`: packs shipped. `line_item_value`: total value in US dollars.
  `pack_price` and `unit_price`: US dollars per pack and per unit.
- `manufacturing_site`: the plant that made the product, often a different company from
  the vendor.
- `first_line_designation`: `true` when the product is a first-line treatment.
- `weight_kilograms`: shipment weight, or `Weight Captured Separately` when it was recorded
  on another line. Use `TRY_CAST(weight_kilograms AS DOUBLE)`.
- `freight_cost_usd`: freight in US dollars, or `Freight Included in Commodity Cost` or
  `Invoiced Separately`. Use `TRY_CAST(freight_cost_usd AS DOUBLE)`.
- `line_item_insurance_usd`: insurance in US dollars, blank when none was charged.

Entities and how they relate: a **shipment line** belongs to a **purchase order**
(`po_so_number`) and a **project**; it is bought from a **vendor**, made at a
**manufacturing site**, shipped by a **mode** under an **Incoterm** to a **country**; the
**product** is identified by brand, molecule, dosage, and form.
DOC

cat >"$cache/incoterms.md" <<'DOC'
# Incoterms used in the shipments table

Incoterms are the standard trade terms that say which party, buyer or seller, arranges
and pays for each leg of a shipment and where the risk of loss passes. The
`vendor_inco_term` column holds the term agreed with each vendor.

- **EXW (Ex Works)**: the seller makes the goods available at its own premises. The buyer
  arranges and pays for everything from that point: loading, export clearance, main
  carriage, insurance, import clearance, and delivery. Maximum obligation on the buyer.
- **FCA (Free Carrier)**: the seller hands the goods, cleared for export, to the carrier the
  buyer nominated at a named place. Risk passes at that hand-over; the buyer pays the main
  carriage.
- **CIP (Carriage and Insurance Paid To)**: the seller pays carriage and insurance to the
  named destination, but risk passes to the buyer once the goods are handed to the first
  carrier. The insurance is for the buyer's benefit.
- **CIF (Cost, Insurance and Freight)**: sea and inland waterway only. The seller pays cost,
  insurance, and freight to the destination port; risk passes when the goods are on board
  the vessel at the port of shipment.
- **DDU (Delivered Duty Unpaid)**: the seller delivers to the named destination; the buyer
  clears import and pays duties. Replaced by DAP in Incoterms 2010 but still common in
  older contracts, including this data.
- **DDP (Delivered Duty Paid)**: the seller delivers to the named destination cleared for
  import with duties paid. Maximum obligation on the seller.
- **N/A - From RDC**: not a trade term. The goods were drawn from a regional distribution
  center that SCMS already owned, so no vendor term applied.

Under EXW and FCA the freight and insurance appear as SCMS costs
(`freight_cost_usd`, `line_item_insurance_usd`); under CIP, CIF, and DDP the vendor's
price already includes them, which the data marks as `Freight Included in Commodity Cost`.
DOC

present="$("$quack" docs -w "$workspace" --json 2>/dev/null || true)"

if [ "$reset" = 1 ] && [ -n "$present" ]; then
  echo "removing every document in '$workspace'"
  printf '%s\n' "$present" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | while read -r id; do
    "$quack" docs -w "$workspace" --delete "$id" >/dev/null
  done
  present=""
fi

filenames="$(printf '%s\n' "$present" | sed -n 's/.*"filename":"\([^"]*\)".*/\1/p')"

for name in shipments.csv shipments-data-dictionary.md incoterms.md importing-into-the-united-states.pdf; do
  if printf '%s\n' "$filenames" | grep -qx "$name"; then
    echo "already in '$workspace': $name"
    continue
  fi
  echo "ingesting $name"
  "$quack" ingest "$cache/$name" -w "$workspace"
done

cat >"$cache/context.md" <<'CTX'
# Logistics workspace

One table, `shipments`: 10,324 line items of HIV drugs and test kits that USAID's Supply
Chain Management System shipped to partner countries, 2006 to 2015. Each line has its
purchase order (`po_so_number`), delivery note (`asn_dn_number`), project, vendor,
manufacturing site, product (brand, molecule, dosage, form), destination country, shipment
mode, Incoterm, milestone dates, quantity, value, freight, and insurance.

Read the data dictionary document before writing SQL: the dates are text
(`try_strptime(col, '%d-%b-%y')`), and weight and freight columns mix numbers with notes
(`TRY_CAST(... AS DOUBLE)`). Money is US dollars.

Documents: the shipments data dictionary; a guide to the Incoterms in the table; and CBP's
"Importing into the United States" on customs entry, commercial invoices, valuation, and
duties. Cite them when the question is about meaning or procedure; use the table when it
is about quantities, costs, dates, vendors, or countries.
CTX

if ! "$quack" context show -w "$workspace" 2>/dev/null | grep -q "Logistics workspace"; then
  echo "setting the workspace context"
  "$quack" context import "$cache/context.md" -w "$workspace"
fi

echo
echo "Workspace '$workspace' is ready. Try:"
echo "  $quack -w $workspace -p \"which vendors shipped the most by value, and what Incoterms did they use?\""
echo "  $quack -w $workspace -p \"how late were ocean shipments to Nigeria on average?\""
echo "  $quack -w $workspace -p \"under EXW, who pays freight and insurance?\" --mode query"
echo "  $quack -w $workspace -p \"what must a commercial invoice show for customs entry?\" --mode query"
