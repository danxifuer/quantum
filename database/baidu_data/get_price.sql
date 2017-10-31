mysqladmin -u root -p create quantum


############### get_price frequence 1d ################
CREATE TABLE `get_price` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `open` DOUBLE(15,10) DEFAULT NULL,
  `high` DOUBLE(15,10) DEFAULT NULL,
  `low` DOUBLE(15,10) DEFAULT NULL,
  `close` DOUBLE(15,10) DEFAULT NULL,
  `ratio` DOUBLE(15,10) DEFAULT NULL,
  `fu_one_ret` DOUBLE(15,10) DEFAULT NULL,
  `volume` float DEFAULT NULL,
  `code` varchar(32) NOT NULL,
  `trade_date` datetime NOT NULL,
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER  TABLE  `get_price`  ADD  INDEX index_name (`code`,  `trade_date`);
