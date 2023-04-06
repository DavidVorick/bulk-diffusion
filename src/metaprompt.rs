#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unused_must_use)]
#![deny(unused_mut)]

use anyhow::{bail, Context, Error, Result};
use no_comment::{languages, IntoWithoutComments as _};
use rand::Rng;
use std::collections::{HashMap, HashSet};

// CONTRIBUTE: An easy contribution would be to improve parsing for the root node, which is
// currently overly strict. It'd be great if each value had a default so that the user didn't have
// to specify every property, and it'd also be great if the user could specify properties in any
// order. We could also extend more of the root node elements to do random selection.

#[derive(Debug)]
enum Selector {
    All,
    One,
    Random(f64),
}

struct Node {
    adds: Vec<(f64, String)>,
    adds_selector: Selector,
    prompts: Vec<(f64, String, Vec<(f64, String)>)>,
    prompts_selector: Selector,
}

/// mut_prompt will randomly add an emphasis or de-emphasis to the prompt according to the
/// frequency provided in the meta-prompt. This type of mutation allows for tuning of the strength
/// of prompts over a large dataset, highlighting when a prompt is better as a stronger or weaker
/// prompt.
fn mut_prompt(
    positive: String,
    negative: String,
    mut_freq: f64,
) -> Result<(String, String), Error> {
    let positives = positive.split(", ");
    let negatives = negative.split(", ");
    let mut positive = "".to_string();
    let mut negative = "".to_string();
    for posi in positives {
        if positive.len() != 0 {
            positive += ", ";
        }
        let rand = rand::thread_rng().gen_range(0.0..1.0);
        if rand < mut_freq {
            let rand = rand::thread_rng().gen_range(0.0..1.0);
            if rand > 0.5 {
                positive += "[";
                positive += posi;
                positive += "]";
            } else {
                positive += "(";
                positive += posi;
                positive += ")";
            }
        } else {
            positive += posi
        }
    }
    for nega in negatives {
        if negative.len() != 0 {
            negative += ", ";
        }
        let rand = rand::thread_rng().gen_range(0.0..1.0);
        if rand < mut_freq {
            let rand = rand::thread_rng().gen_range(0.0..1.0);
            if rand > 0.5 {
                negative += "[";
                negative += nega;
                negative += "]";
            } else {
                negative += "(";
                negative += nega;
                negative += ")";
            }
        } else {
            negative += nega
        }
    }
    Ok((positive, negative))
}

/// generate_prompt will start at the root node and progressively consume nodes until the full
/// prompt is generated.
fn generate_prompt(node_map: &HashMap<String, Node>) -> Result<(String, String), Error> {
    let mut adds_used: HashSet<String> = HashSet::new();
    let mut prompts_used: HashSet<String> = HashSet::new();
    let mut remaining_nodes: Vec<String> = Vec::new();
    remaining_nodes.push("root".to_string());

    // Loop over the remaining nodes and use their contents to build up a prompt.
    let mut positive = "".to_string();
    let mut negative = "".to_string();
    while remaining_nodes.len() > 0 {
        let node = node_map
            .get(&remaining_nodes[0])
            .context(format!("{} is not in node_map", remaining_nodes[0]))?;

        let mut new_adds: HashMap<String, f64> = HashMap::new();
        let mut adds_ordering: Vec<String> = Vec::new();
        if node.prompts.len() == 0 {
            for add in &node.adds {
                new_adds.insert(add.1.to_string(), add.0);
                adds_ordering.push(add.1.to_string());
            }
        } else {
            match node.prompts_selector {
                Selector::All => {
                    for prompt in &node.prompts {
                        let terms = prompt.1.split(",");
                        for term_dirty in terms {
                            let term = term_dirty.trim();
                            if prompts_used.contains(term) {
                                continue;
                            }
                            prompts_used.insert(term.to_string());
                            if term.starts_with('!') {
                                let t = term
                                    .strip_prefix('!')
                                    .context("unable to strip '!' prefix")?;
                                if negative != "" {
                                    negative += ", ";
                                }
                                negative += t;
                            } else {
                                if positive != "" {
                                    positive += ", ";
                                }
                                positive += term;
                            }
                        }
                        for add in &prompt.2 {
                            new_adds.insert(add.1.to_string(), add.0);
                            adds_ordering.push(add.1.to_string());
                        }
                    }
                }
                Selector::One => {
                    // Add up the prompt weights.
                    let mut total_weight = 0.0;
                    for prompt in &node.prompts {
                        total_weight += prompt.0;
                    }

                    // Pick a random prompt.
                    let mut rand = rand::thread_rng().gen_range(0.0..total_weight);
                    let mut index = 0;
                    while index < node.prompts.len() && rand > node.prompts[index].0 {
                        rand -= node.prompts[index].0;
                        index += 1;
                    }

                    let terms = node.prompts[index].1.split(",");
                    for term_dirty in terms {
                        let term = term_dirty.trim();
                        if prompts_used.contains(term) {
                            continue;
                        }
                        prompts_used.insert(term.to_string());
                        if term.starts_with('!') {
                            let t = term
                                .strip_prefix('!')
                                .context("unable to strip '!' prefix")?;
                            if negative != "" {
                                negative += ", ";
                            }
                            negative += t;
                        } else {
                            if positive != "" {
                                positive += ", ";
                            }
                            positive += term;
                        }
                    }

                    for add in &node.prompts[index].2 {
                        new_adds.insert(add.1.to_string(), add.0);
                        adds_ordering.push(add.1.to_string());
                    }
                }
                Selector::Random(f) => {
                    // Add up the prompt weights.
                    let mut total_weight = 0.0;
                    for prompt in &node.prompts {
                        total_weight += prompt.0;
                    }
                    let full_range = total_weight + (total_weight * (1.0 / f));
                    loop {
                        let mut rand = rand::thread_rng().gen_range(0.0..full_range);
                        if rand > total_weight {
                            break;
                        }

                        let mut index = 0;
                        while index < node.prompts.len() && rand > node.prompts[index].0 {
                            rand -= node.prompts[index].0;
                            index += 1;
                        }

                        let terms = node.prompts[index].1.split(",");
                        for term_dirty in terms {
                            let term = term_dirty.trim();
                            if prompts_used.contains(term) {
                                continue;
                            }
                            prompts_used.insert(term.to_string());
                            if term.starts_with('!') {
                                let t = term
                                    .strip_prefix('!')
                                    .context("unable to strip '!' prefix")?;
                                if negative != "" {
                                    negative += ", ";
                                }
                                negative += t;
                            } else {
                                if positive != "" {
                                    positive += ", ";
                                }
                                positive += term;
                            }
                        }

                        for add in &node.prompts[index].2 {
                            new_adds.insert(add.1.to_string(), add.0);
                            adds_ordering.push(add.1.to_string());
                        }
                    }
                }
            }
        }

        match node.adds_selector {
            Selector::All => {
                for add in adds_ordering {
                    if adds_used.contains(&add) {
                        continue;
                    }
                    adds_used.insert(add.to_string());
                    remaining_nodes.push(add);
                }
            }
            Selector::One => {
                let choices = Vec::from_iter(new_adds.iter());
                // Add up the weights.
                let mut total_weight = 0.0;
                for (_, weight) in &choices {
                    total_weight += **weight;
                }

                // Pick a random add.
                let mut rand = rand::thread_rng().gen_range(0.0..total_weight);
                let mut index = 0;
                while index < choices.len() && rand > *choices[index].1 {
                    rand -= choices[index].1;
                    index += 1;
                }

                let choice_dirty = choices[index].0;
                let choice = choice_dirty.trim();
                if adds_used.contains(choice) {
                    continue;
                }
                adds_used.insert(choice.to_string());
                remaining_nodes.push(choice.to_string());
            }
            Selector::Random(f) => {
                let choices = Vec::from_iter(new_adds.iter());
                // Add up the weights.
                let mut total_weight = 0.0;
                for (_, weight) in &choices {
                    total_weight += **weight;
                }
                let full_range = total_weight + (total_weight * (1.0 / f));
                loop {
                    let mut rand = rand::thread_rng().gen_range(0.0..full_range);
                    if rand > total_weight {
                        break;
                    }

                    let mut index = 0;
                    while index < choices.len() && rand > *choices[index].1 {
                        rand -= choices[index].1;
                        index += 1;
                    }

                    let choice_dirty = choices[index].0;
                    let choice = choice_dirty.trim();
                    if adds_used.contains(choice) {
                        continue;
                    }
                    adds_used.insert(choice.to_string());
                    remaining_nodes.push(choice.to_string());
                }
            }
        }

        remaining_nodes.remove(0);
    }

    Ok((positive, negative))
}

impl Node {
    /// default returns the default configuration of a node. The name is 'default', and all of the
    /// fields are empty. The adds_selector defaults to All, and the prompts_selector defaults to
    /// One.
    fn default() -> Node {
        Node {
            adds: Vec::new(),
            adds_selector: Selector::All,
            prompts: Vec::new(),
            prompts_selector: Selector::One,
        }
    }

    /// parse_adds will scroll through the lines of a node description and pull out all the 'adds:'
    /// terms.
    fn parse_adds(&mut self, lines: &Vec<&str>) -> Result<(), Error> {
        let mut reading_adds = false;
        let mut read_adds = false;
        let mut selector = Selector::All;
        for line_dirty in lines {
            let line = line_dirty.trim();
            if trigger_prefix(line) && !adds_prefix(line) {
                reading_adds = false;
                continue;
            } else if adds_prefix(line) && read_adds {
                bail!("can only declare one set of adds");
            } else if adds_prefix(line) {
                reading_adds = true;
                read_adds = true;
                let line_selector = line
                    .strip_prefix("adds:")
                    .context("unable to strip 'adds' prefix")?
                    .trim();
                match line_selector {
                    "one" => selector = Selector::One,
                    "all" => selector = Selector::All,
                    "" => {}
                    _ => {
                        let l = line_selector.trim();
                        let f: f64 = l
                            .parse()
                            .context(format!("invalid selector on adds field: {}", l))?;
                        selector = Selector::Random(f);
                    }
                }
                continue;
            }
            if !reading_adds {
                continue;
            }
            if is_comment(line) {
                continue;
            }

            // At this point, we are reading a line with adds. Parse the weight if there is a
            // weight, and then parse the add.
            if line.contains(',') {
                bail!("anything in an 'adds' line cannot have a ','");
            }
            let final_line: &str;
            let weight: f64;
            if line.contains(':') {
                let parts: Vec<_> = line.split(':').collect();
                if parts.len() != 2 {
                    bail!("'adds' element can only contain one ':', denoting a weight");
                }
                let weight_part = parts[0].trim();
                weight = weight_part
                    .parse()
                    .context("weight in an 'adds' item must be a float64 followed by a colon")?;
                final_line = parts[1].trim();
            } else {
                weight = 1.0;
                final_line = line;
            }
            self.adds.push((weight, final_line.to_string()));
        }
        self.adds_selector = selector;

        Ok(())
    }

    /// parse_prompts will scroll through the lines of a node description and pull out all the
    /// 'prompts' terms.
    fn parse_prompts(&mut self, lines: &Vec<&str>) -> Result<(), Error> {
        let mut reading_prompts = false;
        let mut read_prompts = false;
        let mut selector = Selector::One;
        for line_dirty in lines {
            let line = line_dirty.trim();
            if trigger_prefix(line) && !prompts_prefix(line) {
                reading_prompts = false;
                continue;
            } else if prompts_prefix(line) && read_prompts {
                bail!("can only declare one set of prompts");
            } else if prompts_prefix(line) {
                reading_prompts = true;
                read_prompts = true;
                let line_selector = line
                    .strip_prefix("prompts:")
                    .context("unable to strip 'prompts' prefix")?
                    .trim();
                match line_selector {
                    "one" => selector = Selector::One,
                    "all" => selector = Selector::All,
                    "" => {}
                    _ => {
                        let l = line_selector.trim();
                        let f: f64 = l
                            .parse()
                            .context(format!("invalid selector on prompts field: {}", l))?;
                        selector = Selector::Random(f);
                    }
                }
                continue;
            }
            if !reading_prompts {
                continue;
            }

            // At this point, we are reading a line with prompts. Parse the weight if there is a
            // weight, and then parse the prompts line.
            let final_line: &str;
            let weight: f64;
            if line.contains(':') {
                let parts: Vec<_> = line.split(':').collect();
                if parts.len() != 2 {
                    bail!("'prompts' element can only contain one ':', denoting a weight");
                }
                let weight_part = parts[0].trim();
                weight = weight_part
                    .parse()
                    .context("weight in an 'prompts' item must be a float64 followed by a colon")?;
                final_line = parts[1].trim();
            } else {
                weight = 1.0;
                final_line = line;
            }
            self.prompts
                .push((weight, final_line.to_string(), self.adds.to_vec()));
        }
        self.prompts_selector = selector;

        Ok(())
    }
}

/// parse_name will pull the name out of a node description.
fn parse_name(lines: &Vec<&str>) -> Result<String, Error> {
    let mut name_found = false;
    let mut name = "".to_string();
    for line in lines {
        let trim = line.trim();
        if trim.starts_with("node:") && name_found {
            bail!(
                "cannot have two name fields in a node: {} :: {}",
                name,
                trim
            );
        }
        if trim.starts_with("node:") {
            name = trim
                .strip_prefix("node:")
                .context("unable to strip 'node' prefix")?
                .trim()
                .to_string();
            name_found = true;
        }
    }
    if !name_found {
        bail!("node description does not provide a name: {:?}", lines);
    }
    Ok(name)
}

/// trigger_prefix returns true if the line has a prefix which implies meaning on the following
/// lines.
fn trigger_prefix(line: &str) -> bool {
    line.starts_with("adds:")
        || line.starts_with("root:")
        || line.starts_with("prompts:")
        || line.starts_with("tags:")
        || line.starts_with("node:")
        || line == ""
}

/// adds_prefix returns true if there's a prefix 'adds:' on the line.
fn adds_prefix(line: &str) -> bool {
    line.starts_with("adds:")
}

fn prompts_prefix(line: &str) -> bool {
    line.starts_with("prompts:")
}

fn is_comment(line: &str) -> bool {
    let x = line.trim();
    x.starts_with("//")
}

/// GeneratePromptsResult contains the return value of a call to generate_prompts.
pub struct GeneratePromptsResult {
    /// contains the checkpoint that was parsed from the root node.
    pub checkpoint: String,
    /// is the list of prompts that got generated.
    pub prompts: Vec<String>,
    /// contains the name of the author as presented by the root node.
    pub set_author: String,
    /// contains the name of the set as presented by the root node.
    pub set_name: String,
    /// containes the vae that was parsed from the root node.
    pub vae: String,
}

/// generate_prompts will take metaprompt data and use it to generate 'batch_size' number of
/// prompts that can then be sent to the stable diffusion api.
pub fn generate_prompts(
    metaprompt_bytes: &[u8],
    batch_size: u64,
) -> Result<GeneratePromptsResult, Error> {
    // Convert the metaprompt to a string.
    let metaprompt_str = String::from_utf8(metaprompt_bytes.to_vec())
        .context("unable to parse metaprompt as utf8")?;
    let metaprompt_dirty = metaprompt_str
        .chars()
        .without_comments(languages::rust())
        .collect::<String>();
    let metaprompt = metaprompt_dirty.trim();

    // Split metaprompt into nodes.
    let nodes: Vec<_> = metaprompt.split("\n\n").collect();

    // Parse the root node.
    if nodes.len() < 1 {
        bail!("metaprompt does not start with a valid root node, expecting 11 elements");
    }
    let mut lines: Vec<_> = nodes[0].split("\n").collect();
    if lines.len() < 11 {
        bail!("metaprompt does not start with a valid root node, need at least 11 elements");
    }
    lines[0] = lines[0].trim();
    if lines[0] != "root:" {
        bail!(
            "first node is not a valid root node, 1st line should be 'root:', but is: {}",
            lines[0]
        );
    }
    let set_name_line = lines[1].trim();
    if !set_name_line.starts_with("set name: ") {
        bail!("first node does not have a valid set_name. 2nd line should be a set_name");
    }
    let set_name = set_name_line
        .strip_prefix("set name: ")
        .context("unable to strip prefix")?
        .to_string();
    let set_author_line = lines[2].trim();
    if !set_author_line.starts_with("set author: ") {
        bail!("first node does not have a valid set_author. 3rd line should be a set_author");
    }
    let set_author = set_author_line
        .strip_prefix("set author: ")
        .context("unable to strip prefix")?
        .to_string();
    let checkpoint_line = lines[3].trim();
    if !checkpoint_line.starts_with("checkpoint: ") {
        bail!("first node does not have a valid checkpoint. 4th line should be a checkpoint");
    }
    let checkpoint = checkpoint_line
        .strip_prefix("checkpoint: ")
        .context("unable to strip prefix")?
        .to_string();
    let vae_line = lines[4].trim();
    if !vae_line.starts_with("vae: ") {
        bail!("first node does not have a valid vae. 5th line should be a vae");
    }
    let vae = vae_line
        .strip_prefix("vae: ")
        .context("unable to strip prefix")?
        .to_string();
    let width_line = lines[5].trim();
    if !width_line.starts_with("width: ") {
        bail!("first node does not have a valid width. 6th line should be a width");
    }
    let width = width_line
        .strip_prefix("width: ")
        .context("unable to strip prefix")?;
    let height_line = lines[6].trim();
    if !height_line.starts_with("height: ") {
        bail!("first node does not have a valid height. 7th line should be a heigth");
    }
    let height = height_line
        .strip_prefix("height: ")
        .context("unable to strip prefix")?;
    let upscale_line = lines[7].trim();
    if !upscale_line.starts_with("upscale: ") {
        bail!("first node does not have a valid upscale. 8th line should be a upscale");
    }
    let upscale_str = upscale_line
        .strip_prefix("upscale: ")
        .context("unable to strip prefix")?;
    let upscale = upscale_str
        .parse::<f64>()
        .context("unable to parse upscale")?;
    let sampler_line = lines[8].trim();
    if !sampler_line.starts_with("sampler: ") {
        bail!("first node does not have a valid sampler. 9th line should be a sampler");
    }
    let sampler = sampler_line
        .strip_prefix("sampler: ")
        .context("unable to strip prefix")?;
    let cfg_scale_line = lines[9].trim();
    if !cfg_scale_line.starts_with("cfg: ") {
        bail!("first node does not have a valid cfg. 10th line should be a cfg");
    }
    let cfg_scale = cfg_scale_line
        .strip_prefix("cfg: ")
        .context("unable to strip prefix")?;
    let cfgs: Vec<_> = cfg_scale.split(",").collect();
    let steps_line = lines[10].trim();
    if !steps_line.starts_with("steps: ") {
        bail!("first node does not have a valid step count. 11th line should be a step count");
    }
    let steps = steps_line
        .strip_prefix("steps: ")
        .context("unable to strip prefix")?;
    let stepss: Vec<_> = steps.split(",").collect();
    let mut_freq_line = lines[11].trim();
    if !mut_freq_line.starts_with("mutation frequency: ") {
        bail!("first node does not have a valid mutation frequency");
    }
    let mut_freq_str = mut_freq_line
        .strip_prefix("mutation frequency: ")
        .context("unable to strip prefix")?;
    let mut_freq: f64 = mut_freq_str
        .parse()
        .context("unable to parse mutation frequency")?;

    // Parse each node into the graph.
    let mut node_map: HashMap<String, Node> = HashMap::new();

    // Create the root node.
    let mut root_node = Node::default();
    root_node
        .parse_adds(&lines)
        .context("unable to parse root node adds")?;
    root_node
        .parse_prompts(&lines)
        .context("unable to parse root node prompts")?;
    node_map.insert("root".to_string(), root_node);

    // Parse out all of the remaining nodes.
    for description in &nodes[1..] {
        // Get all the lines of this node and parse out the name.
        let lines: Vec<_> = description.split("\n").collect();
        if lines.len() <= 1 {
            continue;
        }
        let mut node = Node::default();
        let name = parse_name(&lines).context("unable to parse node name")?;
        node.parse_adds(&lines)
            .context("unable to parse node adds")?;
        node.parse_prompts(&lines)
            .context("unable to parse node lines")?;

        if node_map.contains_key(&name) {
            continue;
        }
        node_map.insert(name, node);
    }

    let mut prompts = Vec::new();
    for _ in 0..batch_size {
        let (positive, negative) =
            generate_prompt(&node_map).context("unable to generate prompt")?;
        let (positive, negative) =
            mut_prompt(positive, negative, mut_freq).context("unable to mutate prompt")?;
        let seed = rand::thread_rng().gen_range(0..2100000000);
        let cfg_rng = rand::thread_rng().gen_range(0..cfgs.len());
        let stepss_rng = rand::thread_rng().gen_range(0..stepss.len());

        // Build the prompt based on its conditional elements.
        let mut prompt = format!(
            r###"{{
"seed": {},
"steps": {},
"cfg_scale": {},
"width": {},
"height": {},
"sampler_name": "{}",
"prompt": "{}",
"negative_prompt": "{}",
"batch_size": 8,
"send_images": true"###,
            seed, stepss[stepss_rng], cfgs[cfg_rng], width, height, sampler, positive, negative
        );
        if upscale > 1.0 {
            prompt += &format!(
                r###",
"enable_hr": true,
"hr_scale": {},
"hr_upscaler": "Lanczos",
"denoising_strength": 0.65"###,
                upscale
            );
        }
        prompt += &format!("\n}}");
        prompts.push(prompt);
    }

    Ok(GeneratePromptsResult {
        checkpoint,
        prompts,
        set_author,
        set_name,
        vae,
    })
}
