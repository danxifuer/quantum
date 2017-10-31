

CREATE TABLE `norm_data_across_stock` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `h_o` DOUBLE(15,10) DEFAULT NULL,
  `l_o` DOUBLE(15,10) DEFAULT NULL,
  `c_o` DOUBLE(15,10) DEFAULT NULL,
  `o_c` DOUBLE(15,10) DEFAULT NULL,
  `h_c` DOUBLE(15,10) DEFAULT NULL,
  `l_c` DOUBLE(15,10) DEFAULT NULL,
  `fu_one_ret` DOUBLE(15,10) DEFAULT NULL,
  `volume` DOUBLE(15,10) DEFAULT NULL,
  `code` varchar(32) NOT NULL,
  `trade_date` datetime NOT NULL,
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER  TABLE  `norm_data_across_stock`  ADD  INDEX index_name (`code`,  `trade_date`);

