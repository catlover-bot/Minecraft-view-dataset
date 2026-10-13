# Goal002 documentation correction

The immutable checkpoint uploaded to draft Release `411807868` contains a protocol sentence stating that J changes object key order and JSON whitespace only. That sentence is superseded by:

> J reserializes JSON losslessly: object key order, whitespace and equivalent string-escape or numeric spellings may change, while the entire typed tree is preserved.

For example, an escaped Unicode character may render directly, and an equivalent decimal exponent spelling may change. Exact integer/decimal values and types, boolean/null distinctions, array order, all fields and uncertainty text remain preserved and checked by typed-tree equality and source SHA256.

This is a documentation correction only. No numerical data, implementation, cohort or input identity changed. The original checkpoint archive and its SHA256 remain untouched. The protocol JSON already states the typed-tree preservation rule and needs no correction.
