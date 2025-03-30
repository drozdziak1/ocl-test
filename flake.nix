{
  description = "Realtime 3D printer controller with tablet support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.11";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      system = "x86_64-linux";
      rustVersion = "1.85.0";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      common-deps = with pkgs; [
        cargo-flamegraph
        linuxPackages.perf
        opencl-headers
        openssl
        pkg-config
        rocmPackages_6.clr
        rust-bin.stable."${rustVersion}".default
        rust-analyzer
        zlib
      ];
    in {

      packages.x86_64-linux.hello = nixpkgs.legacyPackages.x86_64-linux.hello;

      defaultPackage.x86_64-linux = self.packages.x86_64-linux.hello;
      devShell."${system}" = pkgs.mkShell rec {
        buildInputs = common-deps;
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

      };
    };
}
