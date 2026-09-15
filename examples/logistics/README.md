# Logistics: a supply chain workspace

One coherent, public-domain dataset that exercises every part of quack at once: a table
to query with SQL, documents in the same vocabulary for vector and keyword search, and a
real entity graph (shipments, purchase orders, vendors, manufacturing sites, products,
countries, Incoterms, shipment modes) for the ontology and knowledge graph.

```bash
make demo-data                                     # load into the workspace named "logistics"
examples/logistics/load.sh WORKSPACE               # load into the workspace named WORKSPACE (created if missing)
examples/logistics/load.sh WORKSPACE --reset       # delete every document in WORKSPACE first, then load
```

Loading is idempotent: a file whose name is already in the workspace is skipped, so
re-running adds nothing. Use `--reset` to start the workspace over.

Then ask:

```bash
quack -w logistics -p "which three vendors shipped the most by value, and which Incoterms did each use?"
quack -w logistics -p "how late were ocean shipments to Nigeria on average?"
quack -w logistics -p "under EXW, who pays freight and insurance?" --mode query
quack -w logistics -p "what must a commercial invoice show for customs entry?" --mode query
```

or open the workspace in the web UI (`make run-server`).

## What is loaded

| Item | What it is | Source |
|---|---|---|
| `shipments` table | 10,324 line items of HIV drugs and test kits USAID's Supply Chain Management System shipped to partner countries, 2006 to 2015: purchase order and delivery note numbers, project, vendor, manufacturing site, product (brand, molecule, dosage, form), country, shipment mode, Incoterm, milestone dates, quantity, value, freight, insurance | [USAID, via data.gov](https://catalog.data.gov/dataset/supply-chain-shipment-pricing-data); fetched from the Internet Archive because data.usaid.gov is gone. Public domain. |
| `shipments-data-dictionary.md` | Every column explained, with the parsing notes the agent needs (dates are text, weight and freight mix numbers with notes) | [documents/](documents/) in this directory |
| `incoterms.md` | The trade terms in the `vendor_inco_term` column: who pays for and bears risk on each leg | [documents/](documents/) in this directory |
| `importing-into-the-united-states.pdf` | CBP's guide for commercial importers: entry, commercial invoices, valuation, duties | [U.S. Customs and Border Protection](https://www.cbp.gov/document/publications/importing-united-states). Public domain. |
| Workspace context | Tells the agent what the table holds, how to parse it, and when to cite which document | [context.md](context.md) |

`load.sh` rewrites the CSV header so columns are SQL identifiers (`po / so #` becomes
`po_so_number`, `freight cost (usd)` becomes `freight_cost_usd`); the data rows are
untouched. Downloads are cached under `~/.cache/quack/examples/logistics/`.

## Why this dataset

- **Keyword search** has exact tokens to find: order numbers like `SCMS-4`, delivery
  notes like `ASN-8`, project codes like `100-CI-T01`, and terms like `EXW`.
- **Vector search** has prose to match by meaning: the customs guide and the Incoterms
  explanations.
- **SQL** has real quantities, money, dates, and enough rows to aggregate by vendor,
  country, mode, and year.
- **The graph** is already in the data: a shipment line belongs to a purchase order and
  a project, is bought from a vendor, made at a manufacturing site, shipped by a mode
  under an Incoterm to a country, and carries a product. The context names those
  relations so ontology induction has a target.

## Adding an example

Make a directory under `examples/` with a `README.md` like this one, a `load.sh` that is
idempotent and takes `[WORKSPACE] [--reset]`, the documents it writes as real files, and a
`context.md`. Only use data whose license allows redistribution, and say what it is.
