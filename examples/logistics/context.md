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
