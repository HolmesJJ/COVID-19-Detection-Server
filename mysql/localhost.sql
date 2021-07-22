-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Jul 18, 2021 at 10:39 AM
-- Server version: 10.4.20-MariaDB
-- PHP Version: 7.3.29

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `covid-19-detection`
--
CREATE DATABASE IF NOT EXISTS `covid-19-detection` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE `covid-19-detection`;

-- --------------------------------------------------------

--
-- Table structure for table `radiographs`
--

CREATE TABLE `radiographs` (
  `id` int(11) NOT NULL,
  `boxes` text DEFAULT NULL,
  `is_marked` int(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `radiographs`
--

INSERT INTO `radiographs` (`id`, `boxes`, `is_marked`) VALUES
(1, NULL, 1),
(2, '[{\'x\': 677.42216, \'y\': 197.97662, \'width\': 867.79767, \'height\': 999.78214}, {\'x\': 1792.69064, \'y\': 402.5525, \'width\': 617.02734, \'height\': 1204.358}]', 0),
(3, NULL, 1),
(4, '[{\'x\': 276.72917, \'y\': 627.42968, \'width\': 910.58859, \'height\': 1655.81519}, {\'x\': 1864.18229, \'y\': 745.22656, \'width\': 875.06262, \'height\': 1535.09888}]', 0),
(5, NULL, 1),
(6, NULL, 1),
(7, NULL, 1),
(8, NULL, 1),
(9, NULL, 1),
(10, NULL, 1);

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `name` text NOT NULL,
  `username` text NOT NULL,
  `password` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `name`, `username`, `password`) VALUES
(1, 'Hu Jiajun', 'holmesjj', '123456'),
(2, 'Zhang Zhiyao', 'bluebell', '123456'),
(3, 'Test1', 'test1', '123456'),
(4, 'Test2', 'test2', '123456');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `radiographs`
--
ALTER TABLE `radiographs`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `radiographs`
--
ALTER TABLE `radiographs`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
