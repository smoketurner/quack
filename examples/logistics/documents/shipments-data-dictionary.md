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
