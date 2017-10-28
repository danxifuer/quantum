
alter table `get_price` add column `future_one_day_returns` DOUBLE(15,10) DEFAULT NULL;
alter table `get_price` add column `pre_two_day_returns` DOUBLE(15,10) DEFAULT NULL;
alter table `get_price` add column `future_two_day_returns` DOUBLE(15,10) DEFAULT NULL;



 CREATE TABLE `norm_data_cross_stock` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `h_o` DOUBLE(15,10) DEFAULT NULL,
  `l_o` DOUBLE(15,10) DEFAULT NULL,
  `c_o` DOUBLE(15,10) DEFAULT NULL,
  `o_c` DOUBLE(15,10) DEFAULT NULL,
  `h_c` DOUBLE(15,10) DEFAULT NULL,
  `l_c` DOUBLE(15,10) DEFAULT NULL,
  `volume` DOUBLE(15,10) DEFAULT NULL,
  `code` varchar(32) NOT NULL,
  `trade_date` datetime NOT NULL,
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER  TABLE  `norm_data_cross_stock`  ADD  INDEX index_name (`code`,  `trade_date`);



################ disprecated below ##################
 CREATE TABLE `pre_two_day_returns` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `close_return` DOUBLE(15,10) DEFAULT NULL,
  `code` varchar(32) DEFAULT NULL,
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `trade_date` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


 CREATE TABLE `future_two_day_returns` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `close_return` DOUBLE(15,10) DEFAULT NULL,
  `code` varchar(32) DEFAULT NULL,
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `trade_date` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


CREATE TABLE `future_one_day_returns` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `close_return` DOUBLE(15,10) DEFAULT NULL,
  `code` varchar(32) DEFAULT NULL,
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `trade_date` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


ALTER  TABLE  `pre_two_day_returns`  ADD  INDEX index_name (`code`,  `trade_date`);
ALTER  TABLE  `future_two_day_returns`  ADD  INDEX index_name (`code`,  `trade_date`);
ALTER  TABLE  `future_one_day_returns`  ADD  INDEX index_name (`code`,  `trade_date`);

