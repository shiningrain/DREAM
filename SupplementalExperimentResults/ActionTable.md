| Object                  | Value                                                                               | Premise                                    |
|-------------------------|-------------------------------------------------------------------------------------|--------------------------------------------|
| augment                 | True, False                                                                         | block type="other"                         |
| block type              | "resnet", "other", "xception", "efficient"                                          | -                                          |
| contrast factor         | "0.1", "0.0", "0.0", "0.1"                                                          | augment=True                               |
| dropout                 | "0.25", "0.0", "0.5"                                                                | -                                          |
| efficientnet version    | "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"                                      | block type="efficient"                     |
| filters                 | "16", "32", "64", "128", "256", "512"                                               | block type="other"                         |
| horizontal flip         | True, False                                                                         | augment=True                               |
| imagenet size           | True, False                                                                         | -                                          |
| kernel size             | "3", "7", "5"                                                                       | block type="other"                         |
| learning rate           | "0.1", "0.01", "0.001", "0.0001", "1e-5", "2e-5"                                    | -                                          |
| max pooling             | True, False                                                                         | block type="other"                         |
| normalize               | True, False                                                                         | -                                          |
| num blocks              | "1", "2", "3"                                                                       | block type="other"                         |
| num layers              | "1", "2"                                                                            | block type="other"                         |
| optimizer               | "sgd", "adam", "adam weight decay", "ndam"                                          | -                                          |
| efficientnet pretrained | False, True                                                                         | block type="efficient"                     |
| resnet pretrained       | False, True                                                                         | block type="resnet"                        |
| xception pretrained     | False, True                                                                         | block type="xception"                      |
| reduction type          | "flatten", "global avg", "global max"                                               | block type="resnet"/"xception"/"efficient" |
| resnet version          | "resnet50", "resnet152", "resnet101", "resnet50 v2", "resnet101 v2", "resnet152 v2" | block type="resnet"                        |
| rotation factor         | "0.1", "0.0", "0.0", "0.1"                                                          | augment=True                               |
| separable               | True, False                                                                         | -                                          |
| efficientnet trainable  | True, False                                                                         | efficientnet pretrained=True               |
| resnet trainable        | True, False                                                                         | resnet pretrained=True                     |
| xception trainable      | True, False                                                                         | xception pretrained=True                   |
| translation factor      | "0.0", "0.1", "0.1", "0.0"                                                          | augment=True                               |
| vertical flip           | True, False                                                                         | augment=True                               |
| zoom factor             | "0.0", "0.1", "0.1", "0.0"                                                          | augment=True                               |
| double step             | True, False                                                                         | trainable=True                             |
| triple step             | True, False                                                                         | Multi step=True                            |
| epoch ratio             | "0.1", "0.2", "0.3", "0.4", "0.5"                                                   | Multi step=True                            |
| lr ratio                | "1.0", "0.1", "0.01"                                                                | Multi step=True                            |
| freeze                  | "all", "no", "bn"                                                                   | Multi step=True                            |
| momentum                | "0.0", "0.1", "0.5", "0.9", "0.99"                                                  | optimizer="sgd"                            |
| end learning rate       | "1e-4", "1e-5", "2e-6", "1e-6", "0.0"                                               | optimizer="adam weight decay"              |
| weight decay rate       | "0.001", "0.005", "0.01", "0.05", "0.1"                                             | optimizer="adam weight decay"              |
| initializer             | "he uniform", "lecun uniform","glorot uniform"                                      | block type="other"                         |
| activation              | "selu", "tanh","relu"                                                               | block type="other"                         |