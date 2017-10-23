
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

