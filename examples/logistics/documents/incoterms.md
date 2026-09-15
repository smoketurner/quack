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
