
 CREATE TABLE `base_info` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `total_num` int(10) unsigned DEFAULT NULL,
  `code` varchar(32) DEFAULT NULL,
  `from_date` datetime DEFAULT NULL,
  `end_date` datetime DEFAULT NULL,
  `close_price_highest` float DEFAULT NULL,
  `close_price_lowest` float DEFAULT NULL,
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


ALTER  TABLE  `base_info`  ADD  INDEX index_name (`code`);
