#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torchaudio

from conette import CoNeTTEConfig, CoNeTTEModel, get_sample_path


class TestInference(TestCase):
    def setUp(self) -> None:
        config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
        model: CoNeTTEModel = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)  # type: ignore

        self.config = config
        self.model = model

    def test_example_1(self) -> None:
        path = get_sample_path()
        outputs = self.model(path)
        candidate = outputs["cands"][0]

        # expected: "rain is pouring down and people are talking in the background"
        self.assertIsInstance(candidate, str)

    def test_example_2(self) -> None:
        path_1 = get_sample_path()
        path_2 = get_sample_path()

        audio_1, sr_1 = torchaudio.load(path_1)  # type: ignore
        audio_2, sr_2 = torchaudio.load(path_2)  # type: ignore

        outputs = self.model([audio_1, audio_2], sr=[sr_1, sr_2])
        candidates = outputs["cands"]

        self.assertIsInstance(candidates, list)
        self.assertEqual(len(candidates), 2)

    def test_example_3(self) -> None:
        path = get_sample_path()
        outputs = self.model(path, task="clotho")
        candidate_cl = outputs["cands"][0]

        outputs = self.model(path, task="audiocaps")
        candidate_ac = outputs["cands"][0]

        self.assertIsInstance(candidate_cl, str)
        self.assertIsInstance(candidate_ac, str)

    def test_forbid_rep_mode(self) -> None:
        path = get_sample_path()
        outputs = self.model(path, task="audiocaps", forbid_rep_mode="none")
        cand = outputs["cands"][0]

        self.assertIsInstance(cand, str)

    def test_tags(self) -> None:
        path = get_sample_path()
        outputs = self.model(path, task="clotho", forbid_rep_mode="none", beam_size=1)
        tags = outputs["tags"]

        assert tags is not None
        self.assertIsInstance(tags, list)


if __name__ == "__main__":
    unittest.main()
