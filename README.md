# To run
* First init databases (__you need to place `db.sql` to the main folder__):
```bash
bash run_db.sh
```
* Then run pretrained model
```bash
bash run_server_pretrained.sh
```

* OR run & train model
```bash
bash run_server.sh
```

* Then use `test.py` to send REST-request from with `request-example.json` data
```bash
bash run_test.sh
```


# Limitations and TODOs
* No kfold, only stratified train-val split was used;
* No batch-processing during inference time; Samples are processed one by one;
* Textual information is not used (there is a reason for that, customers are interested in rewriting texts);
* Probably some places should be checked for side-effects.
